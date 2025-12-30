# Codon Encoder API
# Multi-stage build for minimal image size

FROM python:3.11-slim as builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir --upgrade pip wheel

# Copy requirements and install dependencies
COPY visualizer/requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# =============================================================================
# Production image
# =============================================================================
FROM python:3.11-slim

LABEL maintainer="AI Whisperers <api@ai-whisperers.org>"
LABEL description="Codon Encoder API - Hierarchical codon embeddings"
LABEL version="1.0.0"

# Security: run as non-root user
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /app

# Install dependencies from wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY visualizer/ ./visualizer/
COPY server/ ./server/

# Set ownership
RUN chown -R appuser:appuser /app

USER appuser

# Environment configuration
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV CODON_MODEL_PATH=/app/server/model/codon_encoder.pt

# Expose port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8765/api/metadata')" || exit 1

# Default command
WORKDIR /app/visualizer
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8765"]
