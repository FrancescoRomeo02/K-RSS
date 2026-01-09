# K-RSS Scraper Service
# Lightweight image for YouTube RSS scraping

FROM python:3.11-slim

LABEL maintainer="K-RSS Team"
LABEL description="YouTube RSS Scraper for K-RSS Recommendation System"

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements/scraper.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Create necessary directories
RUN mkdir -p /app/data/raw /app/data/channels

# Default command
CMD ["python", "scraper.py", "--help"]
