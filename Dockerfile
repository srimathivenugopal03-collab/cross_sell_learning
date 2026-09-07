# Multi-stage production-ready Dockerfile using uv for fast and reproducible builds

# Stage 1: Builder
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set virtual environment path
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy project definition and lockfiles
COPY pyproject.toml requirements.txt ./

# Install dependencies into virtualenv using uv
RUN uv venv /opt/venv && \
    uv pip install --no-cache -r requirements.txt

# Stage 2: Final Runtime Image
FROM python:3.11-slim AS runtime

WORKDIR /app

# Create non-root app user for container security
RUN groupadd -r appuser && useradd -r -g appuser -d /app appuser

# Copy virtualenv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Copy application source code and resources
COPY counter/ /app/counter/
COPY resources/ /app/resources/
COPY pyproject.toml /app/

# Create tmp directories with proper permissions
RUN mkdir -p /app/tmp/debug && chown -R appuser:appuser /app

USER appuser

# Expose API port
EXPOSE 5000

# Healthcheck probe
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Start the Flask web application
CMD ["python", "-m", "counter.entrypoints.webapp"]
