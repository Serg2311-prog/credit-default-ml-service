"""Flask API entrypoint for online predictions."""

from __future__ import annotations

import os
from http import HTTPStatus
from typing import Any

from flask import Flask, jsonify, request

from app.model_handler import ModelHandler, ModelLoadError, PredictionError


DEFAULT_MODEL_PATH = os.getenv("MODEL_PATH", "models/artifacts/model.pkl")


def create_app(model_path: str | None = None) -> Flask:
    app = Flask(__name__)

    selected_model_path = model_path or DEFAULT_MODEL_PATH
    app.config["MODEL_PATH"] = selected_model_path

    try:
        app.config["MODEL_HANDLER"] = ModelHandler(selected_model_path)
    except ModelLoadError as exc:
        app.logger.exception("Model initialization failed.")
        app.config["MODEL_HANDLER_ERROR"] = str(exc)
        app.config["MODEL_HANDLER"] = None

    @app.get("/health")
    def health() -> tuple[Any, int]:
        model_handler = app.config.get("MODEL_HANDLER")
        model_error = app.config.get("MODEL_HANDLER_ERROR")
        healthy = model_handler is not None and model_error is None

        response = {
            "status": "ok" if healthy else "degraded",
            "healthy": healthy,
            "model_version": model_handler.model_version if model_handler else "unavailable",
        }
        if model_error:
            response["error"] = model_error
        return jsonify(response), HTTPStatus.OK if healthy else HTTPStatus.SERVICE_UNAVAILABLE

    @app.post("/predict")
    def predict() -> tuple[Any, int]:
        model_handler: ModelHandler | None = app.config.get("MODEL_HANDLER")
        if model_handler is None:
            return (
                jsonify(
                    {
                        "error": "Model is unavailable.",
                        "model_version": "unavailable",
                    }
                ),
                HTTPStatus.SERVICE_UNAVAILABLE,
            )

        payload = request.get_json(silent=True)
        if payload is None or not isinstance(payload, dict):
            return (
                jsonify(
                    {
                        "error": "Request body must be a JSON object.",
                        "model_version": model_handler.model_version,
                    }
                ),
                HTTPStatus.BAD_REQUEST,
            )

        try:
            result = model_handler.predict(payload)
        except PredictionError as exc:
            return (
                jsonify({"error": str(exc), "model_version": model_handler.model_version}),
                HTTPStatus.BAD_REQUEST,
            )

        return (
            jsonify(
                {
                    "prediction": result.prediction,
                    "probability": result.probability,
                    "model_version": result.model_version,
                }
            ),
            HTTPStatus.OK,
        )

    return app


app = create_app()

