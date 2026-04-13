"""Model training script for credit default prediction."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_NAMES = [
    "credit_limit",
    "age",
    "bill_amount",
    "payment_amount",
    "late_payments_6m",
]
RANDOM_STATE = 42


def _build_synthetic_dataset(rows: int = 4000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_STATE)

    credit_limit = rng.normal(5000, 1800, rows).clip(500, 15000)
    age = rng.integers(21, 75, rows)
    bill_amount = rng.normal(2200, 1200, rows).clip(0, 12000)
    payment_amount = rng.normal(1600, 900, rows).clip(0, 12000)
    late_payments_6m = rng.integers(0, 7, rows)

    # Synthetic but realistic rule: high utilization and many delays increase default risk.
    utilization = bill_amount / np.maximum(credit_limit, 1.0)
    payment_ratio = payment_amount / np.maximum(bill_amount, 1.0)
    linear_term = (
        -1.4
        + 2.1 * utilization
        - 1.8 * payment_ratio
        + 0.22 * late_payments_6m
        + 0.015 * (30 - np.minimum(age, 30))
    )
    linear_term = np.clip(linear_term, -20, 20)
    probabilities = 1.0 / (1.0 + np.exp(-linear_term))
    defaulted = rng.binomial(1, probabilities)

    return pd.DataFrame(
        {
            "credit_limit": credit_limit,
            "age": age,
            "bill_amount": bill_amount,
            "payment_amount": payment_amount,
            "late_payments_6m": late_payments_6m,
            "defaulted": defaulted,
        }
    )


def train_and_save_model(output_dir: str = "models/artifacts") -> tuple[str, dict[str, float]]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    data = _build_synthetic_dataset()
    x = data[FEATURE_NAMES]
    y = data["defaulted"]

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=600, random_state=RANDOM_STATE)),
        ]
    )
    model_pipeline.fit(x_train, y_train)

    y_pred = model_pipeline.predict(x_test)
    y_prob = model_pipeline.predict_proba(x_test)[:, 1]

    metrics = {
        "f1": round(float(f1_score(y_test, y_pred)), 4),
        "recall": round(float(recall_score(y_test, y_pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
    }

    model_version = datetime.now(timezone.utc).strftime("v1-%Y%m%dT%H%M%SZ")
    artifact_payload = {
        "model": model_pipeline,
        "feature_names": FEATURE_NAMES,
        "model_version": model_version,
    }

    model_file = output_path / "model.pkl"
    joblib.dump(artifact_payload, model_file)

    metadata_file = output_path / "metadata.json"
    metadata_file.write_text(
        json.dumps({"model_version": model_version, "metrics": metrics}, indent=2),
        encoding="utf-8",
    )

    return model_version, metrics


if __name__ == "__main__":
    version, train_metrics = train_and_save_model()
    print(f"Training complete. Model version: {version}")
    print(f"Metrics: {train_metrics}")

