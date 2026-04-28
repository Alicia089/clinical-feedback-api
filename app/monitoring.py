import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger("clinical-feedback-api.monitoring")


def log_prediction(
    request_id: str,
    text_length: int,
    label: str,
    confidence: float,
    latency_ms: float,
    source: str,
) -> None:
    record = {
        "event": "prediction",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "text_length": text_length,
        "label": label,
        "confidence": confidence,
        "latency_ms": latency_ms,
        "source": source,
    }
    logger.info(json.dumps(record))


def log_model_load(model_path: str, device: str, load_time_ms: float) -> None:
    record = {
        "event": "model_load",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_path": model_path,
        "device": device,
        "load_time_ms": load_time_ms,
    }
    logger.info(json.dumps(record))
