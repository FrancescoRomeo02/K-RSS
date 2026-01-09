# K-RSS Full Pipeline Runner
# Complete environment for running the full recommendation pipeline

FROM python:3.11-slim

LABEL maintainer="K-RSS Team"
LABEL description="Full Pipeline Runner for K-RSS"

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TRANSFORMERS_CACHE=/app/models/transformers \
    HF_HOME=/app/models/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install all Python dependencies
COPY requirements/ /tmp/requirements/
RUN pip install --no-cache-dir \
    -r /tmp/requirements/scraper.txt \
    -r /tmp/requirements/ai_rm.txt

# Create necessary directories
RUN mkdir -p /app/data /app/models

# Default command
CMD ["python", "-m", "pipeline.run"]
