"""
tests/test_api.py — Integration tests for the Clinical Feedback Triage API.

Run: pytest tests/ -v

Engineering note: These tests mock the classifier to decouple API logic tests
from model loading time. A separate test_model.py covers classifier unit tests
with actual inference on a small model checkpoint.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

# Mock the classifier before importing main to avoid model load on test startup
mock_classifier = MagicMock()
mock_classifier.predict.return_value = {
    "label": "URGENT",
    "confidence": 0.9412,
    "scores": {"URGENT": 0.9412, "NEGATIVE": 0.04, "NEUTRAL": 0.01, "POSITIVE": 0.008},
}

with patch("app.main.classifier", mock_classifier):
    from app.main import app

client = TestClient(app)


# ── Health checks ─────────────────────────────────────────────────────────────
class TestHealthEndpoints:
    def test_health_returns_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    def test_readiness_returns_200_when_model_loaded(self):
        with patch("app.main.classifier", mock_classifier):
            response = client.get("/readiness")
            assert response.status_code == 200


# ── Single classification ─────────────────────────────────────────────────────
class TestClassifyEndpoint:
    def test_classify_returns_correct_schema(self):
        with patch("app.main.classifier", mock_classifier):
            response = client.post(
                "/classify",
                json={"text": "Patient reporting severe chest pain post-op", "source": "patient_survey"},
            )
        assert response.status_code == 200
        data = response.json()
        assert "request_id" in data
        assert "label" in data
        assert "confidence" in data
        assert "urgent" in data
        assert "latency_ms" in data

    def test_classify_urgent_sets_urgent_flag(self):
        with patch("app.main.classifier", mock_classifier):
            response = client.post(
                "/classify",
                json={"text": "Patient reporting severe chest pain"},
            )
        assert response.json()["urgent"] is True

    def test_classify_rejects_short_text(self):
        response = client.post("/classify", json={"text": "ok"})
        assert response.status_code == 422

    def test_classify_rejects_empty_text(self):
        response = client.post("/classify", json={"text": "   "})
        assert response.status_code == 422

    def test_classify_rejects_text_over_2000_chars(self):
        response = client.post("/classify", json={"text": "x" * 2001})
        assert response.status_code == 422

    def test_classify_default_source_is_unknown(self):
        with patch("app.main.classifier", mock_classifier):
            response = client.post("/classify", json={"text": "Pain is manageable today"})
        assert response.json()["source"] == "unknown"


# ── Batch classification ──────────────────────────────────────────────────────
class TestBatchEndpoint:
    def test_batch_returns_all_results(self):
        with patch("app.main.classifier", mock_classifier):
            response = client.post(
                "/classify/batch",
                json={"items": [
                    {"text": "Patient experiencing severe bleeding"},
                    {"text": "Staff was very helpful and kind"},
                ]},
            )
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert "urgent_count" in data
        assert len(data["results"]) == 2

    def test_batch_rejects_over_32_items(self):
        with patch("app.main.classifier", mock_classifier):
            items = [{"text": "Sample feedback text here"} for _ in range(33)]
            response = client.post("/classify/batch", json={"items": items})
        assert response.status_code == 422

    def test_batch_rejects_empty_list(self):
        response = client.post("/classify/batch", json={"items": []})
        assert response.status_code == 422


# ── Response headers ──────────────────────────────────────────────────────────
class TestResponseHeaders:
    def test_latency_header_present(self):
        with patch("app.main.classifier", mock_classifier):
            response = client.post(
                "/classify",
                json={"text": "Everything went smoothly with my surgery"},
            )
        assert "X-Inference-Latency-Ms" in response.headers
