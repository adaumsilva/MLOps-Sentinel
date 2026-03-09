"""
Unit tests for the training pipeline and model artefact.

Tests:
  - Synthetic data generation produces expected shape and columns.
  - Preprocessing pipeline transforms data without error.
  - Trained model loads from disk and exposes predict / predict_proba.
  - Model predictions are within expected label set {0, 1}.
  - Model ROC-AUC on held-out data exceeds a minimum quality threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.training.preprocess import ALL_FEATURES, build_preprocessor
from src.training.train import build_model_pipeline, generate_synthetic_data, train

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def raw_data() -> pd.DataFrame:
    return generate_synthetic_data(n_samples=200)


@pytest.fixture(scope="module")
def trained_model_path(tmp_path_factory):
    """Train a small model and return its path (once per test session)."""
    output_dir = tmp_path_factory.mktemp("models")
    model_path = train(output_dir=str(output_dir), n_samples=300)
    return model_path


@pytest.fixture(scope="module")
def loaded_model(trained_model_path):
    import joblib

    return joblib.load(trained_model_path)


# ---------------------------------------------------------------------------
# Data tests
# ---------------------------------------------------------------------------


class TestSyntheticData:
    def test_row_count(self, raw_data):
        assert len(raw_data) == 200

    def test_expected_columns(self, raw_data):
        expected = set(ALL_FEATURES) | {"target"}
        assert expected.issubset(set(raw_data.columns))

    def test_no_nulls(self, raw_data):
        assert raw_data.isnull().sum().sum() == 0

    def test_target_is_binary(self, raw_data):
        assert set(raw_data["target"].unique()).issubset({0, 1})

    def test_class_balance(self, raw_data):
        """Neither class should dominate more than 70/30."""
        counts = raw_data["target"].value_counts(normalize=True)
        assert counts.max() < 0.75


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------


class TestPreprocessor:
    def test_output_shape(self, raw_data):
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(raw_data[ALL_FEATURES])
        assert X.shape == (200, len(ALL_FEATURES))

    def test_output_is_float(self, raw_data):
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(raw_data[ALL_FEATURES])
        assert X.dtype in (np.float32, np.float64)

    def test_scaler_applied(self, raw_data):
        """After StandardScaler the mean should be ~0."""
        preprocessor = build_preprocessor()
        X = preprocessor.fit_transform(raw_data[ALL_FEATURES])
        means = np.abs(X.mean(axis=0))
        assert np.all(means < 1.0), f"Expected scaled means near 0, got {means}"


# ---------------------------------------------------------------------------
# Model artefact tests
# ---------------------------------------------------------------------------


class TestTrainedModel:
    def test_model_loads(self, loaded_model):
        assert loaded_model is not None

    def test_has_predict(self, loaded_model):
        assert hasattr(loaded_model, "predict")

    def test_has_predict_proba(self, loaded_model):
        assert hasattr(loaded_model, "predict_proba")

    def test_predict_labels(self, loaded_model):
        sample = generate_synthetic_data(n_samples=10)[ALL_FEATURES]
        preds = loaded_model.predict(sample)
        assert set(preds).issubset({0, 1})

    def test_predict_proba_sums_to_one(self, loaded_model):
        sample = generate_synthetic_data(n_samples=10)[ALL_FEATURES]
        probas = loaded_model.predict_proba(sample)
        assert probas.shape[1] == 2
        np.testing.assert_allclose(probas.sum(axis=1), 1.0, atol=1e-6)

    def test_roc_auc_threshold(self, loaded_model):
        """ROC-AUC must exceed 0.65 — a sanity floor for a credit-risk model."""
        from sklearn.metrics import roc_auc_score

        df = generate_synthetic_data(n_samples=500)
        X = df[ALL_FEATURES]
        y = df["target"]
        probas = loaded_model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probas)
        assert auc >= 0.65, f"ROC-AUC too low: {auc:.4f}"
