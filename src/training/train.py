"""
Training entry point for MLOps-Sentinel — Credit Risk Assessment.

Usage:
    python -m src.training.train [--output-dir models/] [--n-samples 5000]

The script:
1. Generates (or loads) training data.
2. Runs the preprocessing pipeline.
3. Trains a GradientBoostingClassifier (sklearn) wrapped in a full Pipeline.
4. Evaluates on a held-out test split.
5. Persists the artefact with a UTC timestamp for version traceability.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Ensure project root is on sys.path when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.training.preprocess import (  # noqa: E402
    ALL_FEATURES,
    build_preprocessor,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%SZ",
)
log = logging.getLogger("sentinel.train")

# ---------------------------------------------------------------------------
# Synthetic data generation (replace with real data loader in production)
# ---------------------------------------------------------------------------

RANDOM_SEED = 42


def generate_synthetic_data(n_samples: int = 5_000) -> pd.DataFrame:
    """
    Generate a synthetic credit-risk dataset.
    Target = 1 → high-risk (default likely), 0 → low-risk.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    age = rng.integers(21, 70, n_samples).astype(float)
    annual_income = rng.lognormal(mean=10.8, sigma=0.5, size=n_samples)
    loan_amount = rng.uniform(1_000, 40_000, n_samples)
    credit_score = rng.integers(300, 850, n_samples).astype(float)
    employment_years = rng.integers(0, 30, n_samples).astype(float)
    debt_to_income_ratio = rng.uniform(0.0, 0.9, n_samples)
    num_open_accounts = rng.integers(1, 20, n_samples).astype(float)
    num_derogatory_marks = rng.integers(0, 10, n_samples).astype(float)
    home_ownership = rng.choice(["RENT", "OWN", "MORTGAGE"], n_samples)
    loan_purpose = rng.choice(
        ["DEBT_CONSOLIDATION", "CREDIT_CARD", "HOME_IMPROVEMENT", "OTHER"], n_samples
    )

    # Deterministic risk signal based on domain logic
    risk_score = (
        -0.015 * credit_score
        + 0.8 * debt_to_income_ratio
        + 0.3 * num_derogatory_marks
        - 0.005 * annual_income / 1_000
        + 0.5 * (loan_amount / annual_income)
        + rng.normal(0, 0.3, n_samples)
    )
    target = (risk_score > risk_score.mean()).astype(int)

    return pd.DataFrame(
        {
            "age": age,
            "annual_income": annual_income,
            "loan_amount": loan_amount,
            "credit_score": credit_score,
            "employment_years": employment_years,
            "debt_to_income_ratio": debt_to_income_ratio,
            "num_open_accounts": num_open_accounts,
            "num_derogatory_marks": num_derogatory_marks,
            "home_ownership": home_ownership,
            "loan_purpose": loan_purpose,
            "target": target,
        }
    )


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    classifier = GradientBoostingClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_SEED,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("classifier", classifier)])


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train(output_dir: str = "models", n_samples: int = 5_000) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    log.info("Generating synthetic training data (%d samples)…", n_samples)
    df = generate_synthetic_data(n_samples)
    df.to_csv(Path("data") / "credit_risk_synthetic.csv", index=False)
    log.info("Data written to data/credit_risk_synthetic.csv")

    X = df[ALL_FEATURES]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    log.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    model = build_model_pipeline()
    log.info("Training GradientBoostingClassifier…")
    model.fit(X_train, y_train)

    # --- Evaluation ---
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    log.info("ROC-AUC: %.4f", roc_auc)
    log.info("\n%s", classification_report(y_test, y_pred))

    # --- Versioned artefact ---
    version = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = output_path / f"model_{version}.joblib"
    latest_path = output_path / "model_latest.joblib"

    joblib.dump(model, model_path)
    # Atomic symlink replacement for "latest" pointer
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    # On Windows use a copy; on POSIX use a symlink
    try:
        latest_path.symlink_to(model_path.name)
    except (OSError, NotImplementedError):
        import shutil
        shutil.copy2(model_path, latest_path)

    log.info("Model saved → %s", model_path)
    log.info("Latest pointer → %s", latest_path)

    # --- Persist metrics alongside the artefact ---
    metrics = {
        "version": version,
        "roc_auc": round(roc_auc, 6),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "classification_report": report,
    }
    metrics_path = output_path / f"metrics_{version}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    log.info("Metrics saved → %s", metrics_path)

    return latest_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MLOps-Sentinel training pipeline")
    parser.add_argument("--output-dir", default="models", help="Directory to save model artefacts")
    parser.add_argument("--n-samples", type=int, default=5_000, help="Synthetic dataset size")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(output_dir=args.output_dir, n_samples=args.n_samples)
