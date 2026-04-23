FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install python dependencies first to cache the layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# CRITICAL STEP: Pre-download the massive 1GB Davlan transformer natively into the Docker Image.
# If we don't do this during build, the API will try to pull it from HuggingFace
# upon every single server boot/restart!
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Davlan/afro-xlmr-base')"

# Copy the API logic and the optimized semi-supervised BERTopic model artifacts
COPY api.py .
COPY bertopic_semi_supervised ./bertopic_semi_supervised

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
