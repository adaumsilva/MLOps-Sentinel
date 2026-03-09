"""
MLOps-Sentinel — FastAPI inference service.

Endpoints:
  GET  /health    — liveness + model-loaded check
  POST /predict   — credit-risk classification
  GET  /metrics   — Prometheus scrape endpoint
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from src.api.metrics import PREDICTION_COUNT, PREDICTION_LATENCY, REQUEST_COUNT
from src.api.schemas import CreditRiskInput, HealthResponse, PredictionResponse

log = logging.getLogger("sentinel.api")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)

# ---------------------------------------------------------------------------
# Model state (loaded once at startup — Twelve-Factor: stateless processes)
# ---------------------------------------------------------------------------

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model_latest.joblib"))

_state: dict[str, Any] = {"model": None, "version": "unknown"}


def _load_model() -> None:
    if not MODEL_PATH.exists():
        log.warning("Model artefact not found at %s. Run `make train` first.", MODEL_PATH)
        return
    _state["model"] = joblib.load(MODEL_PATH)
    _state["version"] = MODEL_PATH.stem
    log.info("Model loaded from %s (version=%s)", MODEL_PATH, _state["version"])


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    log.info("Shutting down MLOps-Sentinel API.")


app = FastAPI(
    title="MLOps-Sentinel",
    description="Production-grade Credit Risk Assessment API with observability.",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware — record every request in Prometheus
# ---------------------------------------------------------------------------


@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start

    # Only instrument application routes (skip /metrics itself to avoid noise)
    if request.url.path != "/metrics":
        REQUEST_COUNT.labels(
            endpoint=request.url.path,
            http_status=str(response.status_code),
        ).inc()

    return response


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
def health() -> HealthResponse:
    """Liveness probe — used by Kubernetes / load-balancers."""
    REQUEST_COUNT.labels(endpoint="/health", http_status="200").inc()
    return HealthResponse(status="ok", model_loaded=_state["model"] is not None)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: CreditRiskInput) -> PredictionResponse:
    """
    Run credit-risk inference.

    - **prediction**: 1 = high-risk, 0 = low-risk
    - **probability_high_risk**: model confidence (0–1)
    - **model_version**: artefact identifier loaded at startup
    """
    if _state["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. The service is not ready to serve predictions.",
        )

    # Build a single-row DataFrame to preserve feature names (avoids sklearn warnings)
    input_df = pd.DataFrame([payload.model_dump()])

    with PREDICTION_LATENCY.time():
        prediction: int = int(_state["model"].predict(input_df)[0])
        probability: float = float(_state["model"].predict_proba(input_df)[0][1])

    outcome_label = "high_risk" if prediction == 1 else "low_risk"
    PREDICTION_COUNT.labels(outcome=outcome_label).inc()

    log.info(
        "Prediction: %s (p=%.4f) | credit_score=%d dti=%.2f",
        outcome_label,
        probability,
        payload.credit_score,
        payload.debt_to_income_ratio,
    )

    return PredictionResponse(
        prediction=prediction,
        probability_high_risk=round(probability, 6),
        model_version=_state["version"],
    )


@app.get("/metrics", tags=["Operations"], include_in_schema=False)
def metrics() -> Response:
    """Prometheus scrape endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
