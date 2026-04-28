"""
FeedbackClassifier — Clinical Sentiment Model

Engineering decisions:
  - Base model: distilbert-base-uncased
      WHY: DistilBERT is 40% smaller and 60% faster than BERT-base while
      retaining 97% of its language understanding (Sanh et al., 2019).
      For a real-time API, this latency advantage is non-negotiable.
      A full BERT or RoBERTa model would add ~80ms per inference — too slow
      for synchronous patient-facing workflows.

  - 4-class schema: URGENT / NEGATIVE / NEUTRAL / POSITIVE
      WHY: Standard 3-class sentiment (pos/neg/neu) doesn't capture clinical
      urgency. An URGENT label maps to safety-critical keywords (pain, bleeding,
      fever, fall, allergic reaction) and triggers a separate alerting path
      in the CaseReady workflow, distinct from general dissatisfaction.

  - Inference: torch.no_grad() + model.eval() + half-precision on GPU
      WHY: Disabling gradient tracking reduces memory overhead by ~50% during
      inference. Half-precision (FP16) halves GPU memory and speeds up matmul
      on modern NVIDIA hardware with no meaningful accuracy loss for inference.

  - Thread safety: Model loaded once at startup via FastAPI lifespan,
      not re-instantiated per request.
      WHY: Transformer model loading is expensive (~1-3s). Re-loading per
      request would make the API unusable. The singleton pattern with lifespan
      management is the correct production approach.
"""

import os
import logging
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger("clinical-feedback-api.model")

# ── Label mapping ─────────────────────────────────────────────────────────────
# This matches the fine-tuning label schema.
# In production, this is serialized into the model's config.json.
ID2LABEL = {0: "URGENT", 1: "NEGATIVE", 2: "NEUTRAL", 3: "POSITIVE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# Env-configurable model path — swap between local fine-tuned and HuggingFace Hub
MODEL_PATH = os.getenv("MODEL_PATH", "distilbert-base-uncased")
MAX_LENGTH = int(os.getenv("MAX_TOKEN_LENGTH", "256"))


class FeedbackClassifier:
    """
    Thin wrapper around a HuggingFace SequenceClassification model.
    Handles device selection, tokenization, and inference.
    """

    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.device = self._select_device()
        logger.info(f"FeedbackClassifier will run on: {self.device}")

    def _select_device(self) -> torch.device:
        """
        Device selection priority: CUDA GPU > Apple MPS > CPU.
        WHY MPS: Allows local Mac development to benefit from GPU acceleration
        without requiring a cloud instance during development cycles.
        """
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def load(self):
        """
        Load tokenizer and model. Called once at application startup.

        In a production ML system, this would load from:
          1. A local volume mounted into the Docker container (fastest)
          2. S3/Azure Blob at container init (cost-optimized)
          3. HuggingFace Hub (convenient for prototyping)

        MODEL_PATH env var controls which path is used, making the
        container environment-agnostic.
        """
        logger.info(f"Loading tokenizer from: {MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

        logger.info(f"Loading model from: {MODEL_PATH}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_PATH,
            num_labels=4,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,  # Safe during fine-tuning head swap
        )
        self.model.to(self.device)
        self.model.eval()  # Critical: disables dropout for deterministic inference

        # FP16 on CUDA only — MPS and CPU have limited FP16 matmul support
        if self.device.type == "cuda":
            self.model = self.model.half()
            logger.info("Model converted to FP16 for GPU inference")

        logger.info("FeedbackClassifier loaded successfully")

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        """
        Run inference on a single text string.

        Returns:
            {
              "label": "URGENT" | "NEGATIVE" | "NEUTRAL" | "POSITIVE",
              "confidence": float,   # Softmax probability of predicted class
              "scores": dict         # Full probability distribution
            }

        torch.no_grad() decorator: Disables gradient computation for the
        entire call, reducing memory by ~50% vs training mode.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1).squeeze()

        pred_idx = probs.argmax().item()
        label = ID2LABEL[pred_idx]
        confidence = probs[pred_idx].item()

        scores = {ID2LABEL[i]: round(probs[i].item(), 4) for i in range(4)}

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "scores": scores,
        }
