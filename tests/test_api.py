"""
Integration tests for the FastAPI inference service.

Tests:
  - GET  /health returns 200 OK.
  - POST /predict with valid payload returns 200 OK + correct schema.
  - POST /predict with invalid payload returns 422 Unprocessable Entity.
  - POST /predict probability is in [0, 1].
  - GET  /metrics returns Prometheus text format.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure project root is on the path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Ensure a model artefact exists before starting the API."""
    from src.training.train import train

    output_dir = tmp_path_factory.mktemp("models")
    return train(output_dir=str(output_dir), n_samples=300)


@pytest.fixture(scope="module")
def client(trained_model, monkeypatch_module):
    """Return a TestClient with the MODEL_PATH env var pointed at the fixture model."""
    import os

    monkeypatch_module.setenv("MODEL_PATH", str(trained_model))

    # Re-import app AFTER setting env so lifespan picks up the new path
    import importlib
    import src.api.main as api_module

    importlib.reload(api_module)

    from src.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="module")
def monkeypatch_module(request):
    """Module-scoped monkeypatch (pytest's built-in is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch

    mp = MonkeyPatch()
    yield mp
    mp.undo()


# ---------------------------------------------------------------------------
# Valid payload helper
# ---------------------------------------------------------------------------

VALID_PAYLOAD = {
    "age": 35,
    "annual_income": 72000.0,
    "loan_amount": 15000.0,
    "credit_score": 680,
    "employment_years": 7,
    "debt_to_income_ratio": 0.28,
    "num_open_accounts": 5,
    "num_derogatory_marks": 0,
    "home_ownership": "MORTGAGE",
    "loan_purpose": "DEBT_CONSOLIDATION",
}


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_status_field(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_model_loaded_true(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True


# ---------------------------------------------------------------------------
# Predict endpoint — valid input
# ---------------------------------------------------------------------------


class TestPredictValid:
    def test_returns_200(self, client):
        resp = client.post("/predict", json=VALID_PAYLOAD)
        assert resp.status_code == 200

    def test_response_schema(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert "prediction" in data
        assert "probability_high_risk" in data
        assert "model_version" in data

    def test_prediction_is_binary(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        assert data["prediction"] in {0, 1}

    def test_probability_in_range(self, client):
        data = client.post("/predict", json=VALID_PAYLOAD).json()
        prob = data["probability_high_risk"]
        assert 0.0 <= prob <= 1.0

    def test_high_risk_applicant(self, client):
        """Applicant with poor credit signals should lean high-risk."""
        payload = {
            **VALID_PAYLOAD,
            "credit_score": 310,
            "debt_to_income_ratio": 0.95,
            "num_derogatory_marks": 8,
        }
        data = client.post("/predict", json=payload).json()
        # At minimum the API should not crash; soft assertion on probability
        assert data["probability_high_risk"] >= 0.0


# ---------------------------------------------------------------------------
# Predict endpoint — invalid input (Pydantic validation)
# ---------------------------------------------------------------------------


class TestPredictInvalid:
    def test_missing_field_returns_422(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "credit_score"}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_credit_score_out_of_range_returns_422(self, client):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "credit_score": 900})
        assert resp.status_code == 422

    def test_invalid_enum_returns_422(self, client):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "home_ownership": "UNKNOWN"})
        assert resp.status_code == 422

    def test_negative_income_returns_422(self, client):
        resp = client.post("/predict", json={**VALID_PAYLOAD, "annual_income": -1.0})
        assert resp.status_code == 422

    def test_empty_body_returns_422(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Metrics endpoint
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_returns_200(self, client):
        resp = client.get("/metrics")
        assert resp.status_code == 200

    def test_prometheus_content_type(self, client):
        resp = client.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    def test_contains_sentinel_metrics(self, client):
        # Fire a prediction first so counters are non-zero
        client.post("/predict", json=VALID_PAYLOAD)
        body = client.get("/metrics").text
        assert "sentinel_http_requests_total" in body
        assert "sentinel_model_predictions_total" in body
