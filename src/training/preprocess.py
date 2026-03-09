"""
Feature engineering and preprocessing pipeline for Credit Risk Assessment.
Follows sklearn's TransformerMixin protocol for pipeline composability.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Feature definitions — single source of truth shared by training & inference
# ---------------------------------------------------------------------------

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "loan_amount",
    "credit_score",
    "employment_years",
    "debt_to_income_ratio",
    "num_open_accounts",
    "num_derogatory_marks",
]

CATEGORICAL_FEATURES = [
    "home_ownership",   # RENT | OWN | MORTGAGE
    "loan_purpose",     # DEBT_CONSOLIDATION | CREDIT_CARD | HOME_IMPROVEMENT | OTHER
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

HOME_OWNERSHIP_MAP = {"RENT": 0, "OWN": 1, "MORTGAGE": 2}
LOAN_PURPOSE_MAP = {
    "DEBT_CONSOLIDATION": 0,
    "CREDIT_CARD": 1,
    "HOME_IMPROVEMENT": 2,
    "OTHER": 3,
}


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Ordinal-encodes categorical columns using predefined mappings."""

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["home_ownership"] = (
            X["home_ownership"].str.upper().map(HOME_OWNERSHIP_MAP).fillna(-1).astype(int)
        )
        X["loan_purpose"] = (
            X["loan_purpose"].str.upper().map(LOAN_PURPOSE_MAP).fillna(-1).astype(int)
        )
        return X


class FeatureSelector(BaseEstimator, TransformerMixin):
    """Ensures column order and selection are deterministic."""

    def __init__(self, features: list[str] = ALL_FEATURES):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None) -> "FeatureSelector":
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.features].to_numpy(dtype=np.float64)


def build_preprocessor() -> Pipeline:
    """Return the full preprocessing pipeline (encoding → scaling)."""
    return Pipeline(
        steps=[
            ("encoder", CategoricalEncoder()),
            ("selector", FeatureSelector()),
            ("scaler", StandardScaler()),
        ]
    )
