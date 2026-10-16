from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ..app.data import ROOT, load_data
from .linear_regression import LinearRegressionScratch
from .train_linear_regression import CATEGORICAL, NUMERIC, MODEL_PATH as LINEAR_MODEL_PATH, build_training_frame


MODEL_PATH = ROOT / "backend" / "models" / "random_forest_pipeline.joblib"
METADATA_PATH = ROOT / "backend" / "models" / "model_metadata.json"
REPORT_PATH = ROOT / "reports" / "random_forest_metrics.json"
TARGET = "resolution_hours"
FEATURES = CATEGORICAL + NUMERIC


def _pipeline() -> Pipeline:
    preprocessing = ColumnTransformer([
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
    ])
    return Pipeline([
        ("preprocess", preprocessing),
        ("model", RandomForestRegressor(random_state=42, n_jobs=-1)),
    ])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": round(float(mean_absolute_error(actual, predicted)), 4),
        "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 4),
        "r2": round(float(r2_score(actual, predicted)), 4),
    }


def train_model() -> dict:
    started = time.perf_counter()
    data = load_data()
    frame = build_training_frame().sort_values("timestamp").reset_index(drop=True)
    if len(frame) < 30:
        raise ValueError("At least 30 resolved incidents with valid positive resolution hours are required")

    test_start = int(len(frame) * 0.8)
    development, test = frame.iloc[:test_start], frame.iloc[test_start:]
    train_end = int(len(development) * 0.8)
    train = development.iloc[:train_end]
    if len(train) < 30 or test.empty:
        raise ValueError("Chronological split produced too few training or test records")

    search = GridSearchCV(
        estimator=_pipeline(),
        param_grid={
            "model__n_estimators": [120],
            "model__max_depth": [None, 14],
            "model__min_samples_leaf": [3, 8],
        },
        scoring="neg_mean_absolute_error",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )
    search.fit(train[FEATURES], train[TARGET])
    forest = search.best_estimator_
    forest_prediction = np.maximum(0.0, forest.predict(test[FEATURES]))
    actual = test[TARGET].to_numpy(dtype=float)
    median_prediction = np.full(len(test), float(train[TARGET].median()))
    forest_test_metrics = _metrics(actual, forest_prediction)
    median_test_metrics = _metrics(actual, median_prediction)
    quantile_levels = [.05, .25, .5, .75, .95]
    target_quantiles = np.quantile(actual, quantile_levels)
    prediction_quantiles = np.quantile(forest_prediction, quantile_levels)

    linear_test_metrics = None
    if LINEAR_MODEL_PATH.exists():
        linear = LinearRegressionScratch.load(LINEAR_MODEL_PATH)
        linear_test_metrics = _metrics(actual, np.maximum(0.0, linear.predict(test)))

    cv_rows = []
    for index, params in enumerate(search.cv_results_["params"]):
        cv_rows.append({
            "params": {key.removeprefix("model__"): value for key, value in params.items()},
            "mean_cv_mae": round(float(-search.cv_results_["mean_test_score"][index]), 4),
            "std_cv_mae": round(float(search.cv_results_["std_test_score"][index]), 4),
            "rank": int(search.cv_results_["rank_test_score"][index]),
        })

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "model_name": "RandomForestRegressor",
        "artifact_format": 1,
        "artifact_path": str(MODEL_PATH.relative_to(ROOT)),
        "target": TARGET,
        "target_unit": "hours",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(time.perf_counter() - started, 3),
        "dataset": {
            "original_incidents": len(data["incidents"]),
            "eligible_resolved_records": len(frame),
            "excluded_records": len(data["incidents"]) - len(frame),
            "exclusion_rule": "Resolved status, positive numeric resolution_hours, and valid timestamp required.",
            "date_range": [str(frame["timestamp"].min().date()), str(frame["timestamp"].max().date())],
        },
        "split": {
            "method": "Chronological 80/20 test holdout; development period split 80/20 into model-training and validation-only windows.",
            "training_records": len(train),
            "validation_only_records": len(development) - len(train),
            "test_records": len(test),
            "test_start": str(test["timestamp"].min().date()),
            "test_end": str(test["timestamp"].max().date()),
            "leakage_note": "All test records are later than training records. Location IDs are not predictors; location-derived zone/context features are joined many-to-one. Locations may recur across time partitions.",
        },
        "feature_schema": {
            "categorical": CATEGORICAL,
            "numeric": NUMERIC,
            "input_columns": FEATURES,
            "training_category_values": {column: sorted(train[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL},
            "training_numeric_support": {column: {"min": float(train[column].min()), "max": float(train[column].max()), "p99": float(train[column].quantile(.99))} for column in NUMERIC},
            "excluded_columns": ["incident_id", "location_id", "resolved", TARGET, "completion timestamps"],
            "prediction_request_fields": ["location_id", "incident_type", "severity", "reported_by", "reported_injuries"],
            "preprocessing": "Most-frequent imputation and one-hot encoding with unknown-category handling for categoricals; median imputation for numerics; fitted inside the sklearn pipeline on training rows only.",
        },
        "cross_validation": {
            "method": "TimeSeriesSplit with 3 chronological folds on training rows only",
            "selection_metric": "MAE (hours)",
            "best_parameters": {key.removeprefix("model__"): value for key, value in search.best_params_.items()},
            "best_mean_cv_mae": round(float(-search.best_score_), 4),
            "candidates": cv_rows,
        },
        "held_out_test": {
            "random_forest": forest_test_metrics,
            "scratch_linear_regression": linear_test_metrics,
            "median_baseline": median_test_metrics,
            "comparison_note": "All metrics use the exact same chronological held-out test rows. A model is not claimed superior unless its measured metric supports that claim.",
        },
        "prediction_constraint": "Return max(0, raw prediction) hours; model estimates are not guarantees.",
        "target_distribution": {"count": len(actual), "min": round(float(actual.min()), 4), "quantiles": {str(q): round(float(value), 4) for q, value in zip(quantile_levels, target_quantiles)}, "max": round(float(actual.max()), 4)},
        "prediction_distribution": {"count": len(forest_prediction), "min": round(float(forest_prediction.min()), 4), "quantiles": {str(q): round(float(value), 4) for q, value in zip(quantile_levels, prediction_quantiles)}, "max": round(float(forest_prediction.max()), 4), "unique_rounded_2dp": int(len(np.unique(np.round(forest_prediction, 2))))},
    }
    feature_names = forest.named_steps["preprocess"].get_feature_names_out()
    importances = forest.named_steps["model"].feature_importances_
    ranked_importance = sorted(zip(feature_names, importances), key=lambda item: -item[1])
    metadata["feature_importance"] = {
        "top_transformed_features": [{"feature": str(name), "importance": round(float(value), 6)} for name, value in ranked_importance[:25]],
        "by_source_feature": {column: round(float(sum(value for name, value in zip(feature_names, importances) if str(name).startswith((f"categorical__{column}_", f"numeric__{column}")))), 6) for column in FEATURES},
        "note": "Impurity-based Random Forest importances are descriptive, not causal; grouped values sum one-hot columns for each source field.",
    }

    temporary_model = MODEL_PATH.with_suffix(MODEL_PATH.suffix + ".tmp")
    temporary_metadata = METADATA_PATH.with_suffix(METADATA_PATH.suffix + ".tmp")
    joblib.dump(forest, temporary_model)
    probe = joblib.load(temporary_model)
    probe_prediction = np.asarray(probe.predict(test[FEATURES]), dtype=float)
    if probe_prediction.shape != (len(test),) or not np.isfinite(probe_prediction).all():
        temporary_model.unlink(missing_ok=True)
        raise ValueError("Saved Random Forest pipeline failed prediction validation; previous artifact was preserved")
    temporary_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    temporary_model.replace(MODEL_PATH)
    temporary_metadata.replace(METADATA_PATH)
    REPORT_PATH.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


if __name__ == "__main__":
    print(json.dumps(train_model(), indent=2))