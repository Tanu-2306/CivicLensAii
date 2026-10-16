from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd


@dataclass
class FeatureEncoder:
    categorical: list[str]
    numeric: list[str]
    categories: dict[str, list[str]]
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, categorical: list[str], numeric: list[str]) -> "FeatureEncoder":
        categories = {column: sorted(frame[column].fillna("Unknown").astype(str).unique().tolist()) for column in categorical}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in numeric:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            mean = float(values.mean()) if values.notna().any() else 0.0
            scale = float(values.std(ddof=0)) if values.notna().any() else 1.0
            means[column] = mean
            scales[column] = scale if np.isfinite(scale) and scale > 1e-12 else 1.0
        return cls(categorical, numeric, categories, means, scales)

    @property
    def feature_names(self) -> list[str]:
        names = ["intercept"]
        for column, values in self.categories.items():
            names.extend(f"{column}={value}" for value in values[1:])
        names.extend(self.numeric)
        return names

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        columns = [np.ones(len(frame), dtype=float)]
        for column, values in self.categories.items():
            observed = frame[column].fillna("Unknown").astype(str).to_numpy()
            for category in values[1:]:
                columns.append((observed == category).astype(float))
        for column in self.numeric:
            values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(self.means[column]).to_numpy(dtype=float)
            columns.append((values - self.means[column]) / self.scales[column])
        return np.column_stack(columns)

    def to_dict(self) -> dict[str, Any]:
        return {"categorical": self.categorical, "numeric": self.numeric, "categories": self.categories, "means": self.means, "scales": self.scales}

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "FeatureEncoder":
        return cls(values["categorical"], values["numeric"], values["categories"], values["means"], values["scales"])


class LinearRegressionScratch:
    """Ordinary least-squares regression optimized with full-batch gradient descent."""

    def __init__(self, learning_rate: float = 0.03, epochs: int = 5000, patience: int = 250) -> None:
        if learning_rate <= 0 or epochs < 1 or patience < 1:
            raise ValueError("learning_rate, epochs, and patience must be positive")
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.patience = patience
        self.encoder: FeatureEncoder | None = None
        self.weights: np.ndarray | None = None
        self.target_mean = 0.0
        self.target_scale = 1.0
        self.training_loss: list[float] = []
        self.validation_loss: list[float] = []

    @staticmethod
    def loss_and_gradient(features: np.ndarray, targets: np.ndarray, weights: np.ndarray) -> tuple[float, np.ndarray]:
        errors = features @ weights - targets
        return float(np.mean(errors**2)), (2.0 / len(features)) * (features.T @ errors)

    def fit(self, training_frame: pd.DataFrame, targets: np.ndarray, validation_frame: pd.DataFrame | None = None, validation_targets: np.ndarray | None = None, categorical: list[str] | None = None, numeric: list[str] | None = None, on_epoch: Callable[[dict[str, float | int]], None] | None = None) -> "LinearRegressionScratch":
        categorical = categorical or []
        numeric = numeric or []
        target_values = np.asarray(targets, dtype=float)
        if len(training_frame) != len(target_values) or len(target_values) < 2 or not np.isfinite(target_values).all():
            raise ValueError("Training data must have at least two rows and finite targets")
        self.encoder = FeatureEncoder.fit(training_frame, categorical, numeric)
        x_train = self.encoder.transform(training_frame)
        self.target_mean = float(target_values.mean())
        target_std = float(target_values.std())
        self.target_scale = target_std if target_std > 1e-12 else 1.0
        y_train = (target_values - self.target_mean) / self.target_scale
        x_validation = self.encoder.transform(validation_frame) if validation_frame is not None else x_train
        y_validation_values = np.asarray(validation_targets if validation_targets is not None else targets, dtype=float)
        y_validation = (y_validation_values - self.target_mean) / self.target_scale
        if len(x_validation) != len(y_validation):
            raise ValueError("Validation features and targets must have matching lengths")

        weights = np.zeros(x_train.shape[1], dtype=float)
        best_weights = weights.copy()
        best_validation = float("inf")
        stale_epochs = 0
        self.training_loss = []
        self.validation_loss = []
        for epoch in range(1, self.epochs + 1):
            _, gradient = self.loss_and_gradient(x_train, y_train, weights)
            if not np.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient at epoch {epoch}")
            weights -= self.learning_rate * gradient
            training_loss, _ = self.loss_and_gradient(x_train, y_train, weights)
            validation_loss, _ = self.loss_and_gradient(x_validation, y_validation, weights)
            training_mse = training_loss * self.target_scale**2
            validation_mse = validation_loss * self.target_scale**2
            if not np.isfinite([training_mse, validation_mse]).all():
                raise FloatingPointError(f"Non-finite loss at epoch {epoch}")
            self.training_loss.append(float(training_mse))
            self.validation_loss.append(float(validation_mse))
            if on_epoch is not None:
                on_epoch({"epoch": epoch, "training_mse": float(training_mse), "validation_mse": float(validation_mse)})
            if validation_loss < best_validation - 1e-9:
                best_validation = validation_loss
                best_weights = weights.copy()
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break
        self.weights = best_weights
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.encoder is None or self.weights is None:
            raise RuntimeError("Model has not been trained or loaded")
        standardized = self.encoder.transform(frame) @ self.weights
        return standardized * self.target_scale + self.target_mean

    def save(self, path: str | Path) -> None:
        if self.encoder is None or self.weights is None:
            raise RuntimeError("Cannot save an untrained model")
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {"format": 2, "learning_rate": self.learning_rate, "epochs": self.epochs, "patience": self.patience, "encoder": self.encoder.to_dict(), "weights": self.weights.tolist(), "target_mean": self.target_mean, "target_scale": self.target_scale, "training_loss": self.training_loss, "validation_loss": self.validation_loss}
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        restored = self.load(temporary)
        if not np.isfinite(restored.weights).all():
            temporary.unlink(missing_ok=True)
            raise ValueError("Refusing to replace model with invalid parameters")
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path) -> "LinearRegressionScratch":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") not in {1, 2}:
            raise ValueError("Unsupported linear regression model format")
        model = cls(payload["learning_rate"], payload["epochs"], payload["patience"])
        model.encoder = FeatureEncoder.from_dict(payload["encoder"])
        model.weights = np.asarray(payload["weights"], dtype=float)
        model.target_mean = float(payload["target_mean"])
        model.target_scale = float(payload["target_scale"])
        model.training_loss = payload.get("training_loss", [])
        model.validation_loss = payload.get("validation_loss", [])
        if model.weights.shape != (len(model.encoder.feature_names),) or not np.isfinite(model.weights).all():
            raise ValueError("Invalid weights in linear regression model")
        return model