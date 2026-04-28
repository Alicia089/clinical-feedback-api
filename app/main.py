"""
Clinical Feedback Triage API
Komereglissade / CaseReady — Alicia Eradiri

Classifies patient and surgical center feedback into:
  URGENT | NEGATIVE | NEUTRAL | POSITIVE

Designed for Ambulatory Surgery Centers to surface high-priority
feedback before it escalates. Built with production MLOps principles:
containerized, monitored, and cloud-deployable.
"""

import time
import uuid
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator

from app.schemas import FeedbackRequest, FeedbackResponse, BatchFeedbackRequest, BatchFeedbackResponse
from app.model import FeedbackClassifier
from app.monitoring import log_prediction

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("clinical-feedback-api")

# ── Model lifecycle ───────────────────────────────────────────────────────────
classifier: FeedbackClassifier | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    global classifier
    logger.info("Loading FeedbackClassifier...")
    classifier = FeedbackClassifier()
    classifier.load()
    logger.info("Model ready.")
    yield
    logger.info("Shutting down — releasing model.")
    classifier = None


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Clinical Feedback Triage API",
    description=(
        "Real-time NLP classification of patient and ASC staff feedback. "
        "Flags URGENT cases for immediate follow-up. Built for CaseReady by Komereglissade."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# Prometheus metrics endpoint at /metrics
Instrumentator().instrument(app).expose(app)


# ── Middleware: request timing ────────────────────────────────────────────────
@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    response.headers["X-Inference-Latency-Ms"] = str(latency_ms)
    return response


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["Ops"])
async def health_check():
    """Kubernetes/ECS liveness probe."""
    return {"status": "healthy", "model_loaded": classifier is not None}


@app.get("/readiness", tags=["Ops"])
async def readiness_check():
    """Kubernetes/ECS readiness probe — fails if model not loaded."""
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    return {"status": "ready"}


@app.post("/classify", response_model=FeedbackResponse, tags=["Inference"])
async def classify_feedback(payload: FeedbackRequest):
    """
    Classify a single piece of clinical feedback.

    Returns a label (URGENT / NEGATIVE / NEUTRAL / POSITIVE),
    a confidence score, and a flag if immediate follow-up is warranted.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    request_id = str(uuid.uuid4())
    start = time.perf_counter()

    try:
        result = classifier.predict(payload.text)
    except Exception as exc:
        logger.error(f"[{request_id}] Inference error: {exc}")
        raise HTTPException(status_code=500, detail="Inference failed")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    log_prediction(
        request_id=request_id,
        text_length=len(payload.text),
        label=result["label"],
        confidence=result["confidence"],
        latency_ms=latency_ms,
        source=payload.source,
    )

    logger.info(
        f"[{request_id}] label={result['label']} "
        f"confidence={result['confidence']:.3f} latency={latency_ms}ms"
    )

    return FeedbackResponse(
        request_id=request_id,
        label=result["label"],
        confidence=result["confidence"],
        urgent=result["label"] == "URGENT",
        latency_ms=latency_ms,
        source=payload.source,
    )


@app.post("/classify/batch", response_model=BatchFeedbackResponse, tags=["Inference"])
async def classify_batch(payload: BatchFeedbackRequest):
    """
    Classify up to 32 feedback items in a single call.
    Useful for bulk ingestion from EHR exports or survey data.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(payload.items) > 32:
        raise HTTPException(status_code=422, detail="Batch size cannot exceed 32")

    results = []
    for item in payload.items:
        request_id = str(uuid.uuid4())
        start = time.perf_counter()
        result = classifier.predict(item.text)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)

        results.append(FeedbackResponse(
            request_id=request_id,
            label=result["label"],
            confidence=result["confidence"],
            urgent=result["label"] == "URGENT",
            latency_ms=latency_ms,
            source=item.source,
        ))

    urgent_count = sum(1 for r in results if r.urgent)

    return BatchFeedbackResponse(
        total=len(results),
        urgent_count=urgent_count,
        results=results,
    )
