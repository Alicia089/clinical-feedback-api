"""
Pydantic schemas — request validation and response serialization.

Engineering decision: Pydantic v2 is used here (FastAPI default) for
its Rust-backed validation speed (~5-17x faster than v1 for complex models).
This matters at scale when processing high-volume feedback ingestion.
"""

from enum import Enum
from typing import Optional, List
from pydantic import BaseModel, Field, field_validator


class FeedbackSource(str, Enum):
    PATIENT_SURVEY = "patient_survey"
    STAFF_NOTE = "staff_note"
    POST_OP_FORM = "post_op_form"
    INTAKE_COMMENT = "intake_comment"
    EHR_EXPORT = "ehr_export"
    UNKNOWN = "unknown"


class SentimentLabel(str, Enum):
    URGENT = "URGENT"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
    POSITIVE = "POSITIVE"


class FeedbackRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=2000,
        description="Raw feedback text from patient or staff",
        examples=["Patient reported severe pain at incision site three hours post-op"],
    )
    source: FeedbackSource = Field(
        default=FeedbackSource.UNKNOWN,
        description="Origin of the feedback for audit logging",
    )

    @field_validator("text")
    @classmethod
    def strip_and_validate(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Feedback text cannot be empty or whitespace only")
        return v


class FeedbackResponse(BaseModel):
    request_id: str = Field(description="UUID for this inference — use for audit trail")
    label: SentimentLabel
    confidence: float = Field(ge=0.0, le=1.0, description="Model confidence [0, 1]")
    urgent: bool = Field(description="True when label == URGENT; triggers downstream alerting")
    latency_ms: float = Field(description="End-to-end inference time in milliseconds")
    source: FeedbackSource


class BatchFeedbackRequest(BaseModel):
    items: List[FeedbackRequest] = Field(
        ...,
        min_length=1,
        max_length=32,
        description="List of feedback items. Hard limit of 32 to protect memory.",
    )


class BatchFeedbackResponse(BaseModel):
    total: int
    urgent_count: int = Field(description="Number of items flagged as URGENT")
    results: List[FeedbackResponse]
