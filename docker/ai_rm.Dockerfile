# K-RSS AI Recommendation Module
# Full ML/DL environment for recommendation system

FROM python:3.11-slim

LABEL maintainer="K-RSS Team"
LABEL description="AI Recommendation Module for K-RSS"

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

# Copy and install Python dependencies
COPY requirements/ai_rm.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create necessary directories
RUN mkdir -p /app/data /app/models/transformers /app/models/huggingface

# Default command
CMD ["python", "-c", "print('AI Recommendation Module ready')"]
