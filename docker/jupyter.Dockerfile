# K-RSS Jupyter Lab Environment
# Full development environment with all dependencies

FROM python:3.11-slim

LABEL maintainer="K-RSS Team"
LABEL description="Jupyter Lab Development Environment for K-RSS"

WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    JUPYTER_ENABLE_LAB=yes

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Copy and install all Python dependencies
COPY requirements/ /tmp/requirements/
RUN pip install --no-cache-dir \
    -r /tmp/requirements/scraper.txt \
    -r /tmp/requirements/ai_rm.txt \
    -r /tmp/requirements/webapp.txt \
    -r /tmp/requirements/dev.txt

# Expose Jupyter port
EXPOSE 8888

# Default command
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
