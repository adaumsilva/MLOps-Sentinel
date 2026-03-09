# =============================================================================
# MLOps-Sentinel — Multi-stage Dockerfile
# Stage 1: builder  — installs dependencies into a venv
# Stage 2: runtime  — copies only the venv + source; no build tools shipped
# =============================================================================

# ---- Stage 1: builder -------------------------------------------------------
FROM python:3.11-slim AS builder

# Security: don't run pip as root in the final image
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Leverage Docker layer cache — copy deps manifest first
COPY requirements.txt .

# Install into an isolated venv so Stage 2 can copy it cleanly
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install -r requirements.txt


# ---- Stage 2: runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Twelve-Factor III: config via env
    MODEL_PATH="models/model_latest.joblib" \
    PORT=8000

# Non-root user for least-privilege principle
RUN addgroup --system sentinel && adduser --system --ingroup sentinel sentinel

WORKDIR /app

# Copy only the venv from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application source
COPY src/ ./src/
COPY models/ ./models/

# Ownership handoff
RUN chown -R sentinel:sentinel /app

USER sentinel

EXPOSE ${PORT}

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
