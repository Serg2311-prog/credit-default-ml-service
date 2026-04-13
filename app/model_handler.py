"""Model loading and inference utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class ModelLoadError(RuntimeError):
    """Raised when model artifacts cannot be loaded."""


class PredictionError(RuntimeError):
    """Raised when inference fails."""


@dataclass(frozen=True)
class PredictionResult:
    prediction: int
    probability: float
    model_version: str


class ModelHandler:
    """Encapsulates model artifact loading and prediction logic."""

    def __init__(self, model_path: str) -> None:
        self.model_path = Path(model_path)
        self._model: Any | None = None
        self._feature_names: list[str] = []
        self._model_version: str = "unknown"
        self._load_model()

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def feature_names(self) -> list[str]:
        return self._feature_names

    def _load_model(self) -> None:
        if not self.model_path.exists():
            raise ModelLoadError(f"Model artifact not found: {self.model_path}")

        try:
            payload = joblib.load(self.model_path)
        except Exception as exc:  # noqa: BLE001
            raise ModelLoadError(f"Failed to load model: {exc}") from exc

        if not isinstance(payload, dict):
            raise ModelLoadError("Invalid model artifact format: expected dictionary payload.")

        self._model = payload.get("model")
        self._feature_names = payload.get("feature_names", [])
        self._model_version = payload.get("model_version", "unknown")

        if self._model is None or not self._feature_names:
            raise ModelLoadError("Model artifact misses required keys: 'model' and 'feature_names'.")

    def predict(self, payload: dict[str, Any]) -> PredictionResult:
        if self._model is None:
            raise PredictionError("Model is not loaded.")

        missing = [feature for feature in self._feature_names if feature not in payload]
        if missing:
            raise PredictionError(f"Missing required features: {missing}")

        row = {}
        for feature in self._feature_names:
            value = payload[feature]
            try:
                row[feature] = float(value)
            except (TypeError, ValueError) as exc:
                raise PredictionError(f"Feature '{feature}' must be numeric.") from exc

        input_df = pd.DataFrame([row], columns=self._feature_names)

        try:
            prediction = int(self._model.predict(input_df)[0])
            probability = float(self._model.predict_proba(input_df)[0][1])
        except Exception as exc:  # noqa: BLE001
            raise PredictionError(f"Inference failed: {exc}") from exc

        return PredictionResult(
            prediction=prediction,
            probability=round(probability, 6),
            model_version=self._model_version,
        )

