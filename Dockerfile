# =============================================================================
# AuditHub - Dockerfile
# =============================================================================
# Multi-stage build for the AuditHub platform.
# Default runtime serves the Streamlit dashboard.

# ---- Stage 1: Base Image ----
FROM python:3.11-slim AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# ---- Stage 2: Dependencies ----
FROM base AS dependencies

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ---- Stage 3: Final Image ----
FROM dependencies AS final

WORKDIR /app

# Create necessary directories
RUN mkdir -p data/raw \
    data/validated \
    data/repaired \
    data/mutated \
    data/processed \
    data/reports \
    artifacts \
    models \
    reports \
    logs \
    mlruns

# Copy project files
COPY . .

# Expose Streamlit and FastAPI ports
EXPOSE 8501
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default command: run Streamlit
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
