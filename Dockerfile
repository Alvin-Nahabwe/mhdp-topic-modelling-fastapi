# ===== Stage 1: Builder =====
# Install dependencies and download model weights in a builder stage
# so build tools (build-essential) don't bloat the production image.
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# Pre-download the 1GB Davlan transformer into HuggingFace cache.
# We use the installed packages from /install to do this.
ENV PYTHONPATH=/install/lib/python3.11/site-packages
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Davlan/afro-xlmr-base')"


# ===== Stage 2: Production =====
FROM python:3.11-slim

# Only curl needed at runtime (for HEALTHCHECK)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder (no build-essential in production)
COPY --from=builder /install /usr/local

# Copy cached HuggingFace model weights from builder
COPY --from=builder /root/.cache/huggingface /root/.cache/huggingface

# Copy application code and model artifacts
COPY api.py .
COPY stop_words.py .
COPY clinical_vocabulary.py .
COPY ordinal_classifier.py .
COPY bertopic_semi_supervised ./bertopic_semi_supervised

# Copy classifier model if available (preferred production model)
COPY classifier_model ./classifier_model

# Copy affect risk models if available
COPY affect_risk_model ./affect_risk_model

# Security: run as non-root user
RUN adduser --disabled-password --gecos '' --no-create-home appuser && \
    # Move HF cache to app-accessible location before switching user
    mkdir -p /home/appuser/.cache && \
    cp -r /root/.cache/huggingface /home/appuser/.cache/ && \
    chown -R appuser:appuser /home/appuser/.cache && \
    rm -rf /root/.cache/huggingface

USER appuser
ENV HF_HOME=/home/appuser/.cache/huggingface

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
