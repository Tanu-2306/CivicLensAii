from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from ..app.data import ROOT, load_data
from .linear_regression import LinearRegressionScratch


MODEL_PATH = ROOT / "models" / "incident_resolution_linear.json"
FOREST_PATH = ROOT / "models" / "incident_resolution_forest.joblib"
REPORT_PATH = ROOT / "reports" / "incident_resolution_linear.json"
CLEANING_REPORT_PATH = ROOT / "reports" / "data_cleaning_report.json"
TRAINING_LOG_PATH = ROOT / "reports" / "linear_regression_training.log"
CATEGORICAL = ["incident_type", "severity", "reported_by", "zone", "road_type", "road_condition", "pedestrian_activity"]
NUMERIC = ["injuries", "hour", "weekday", "month", "population_density_per_km2", "near_school", "near_hospital"]


def build_training_frame() -> pd.DataFrame:
    data = load_data()
    incidents = data["incidents"].copy()
    incidents["resolution_hours"] = pd.to_numeric(incidents["resolution_hours"], errors="coerce")
    valid = incidents["resolved"] & incidents["resolution_hours"].gt(0) & incidents["resolution_hours"].notna() & incidents["timestamp"].notna()
    usable = incidents.loc[valid].merge(data["locations"], on="location_id", how="left", validate="many_to_one")
    timestamp = usable["timestamp"]
    usable["hour"] = timestamp.dt.hour
    usable["weekday"] = timestamp.dt.dayofweek
    usable["month"] = timestamp.dt.month
    for column in ("near_school", "near_hospital"):
        usable[column] = usable[column].astype(float)
    return usable.sort_values("timestamp").reset_index(drop=True)


def build_prediction_frame(payload: dict) -> pd.DataFrame:
    locations = load_data()["locations"]
    location = locations.loc[locations["location_id"] == str(payload["location_id"])]
    if location.empty:
        raise ValueError(f"Unknown location_id: {payload['location_id']}")
    timestamp = pd.to_datetime(payload.get("timestamp"), errors="coerce", utc=True)
    if pd.isna(timestamp):
        timestamp = pd.Timestamp.now(tz="UTC")
    row = location.iloc[0].to_dict()
    row.update({
        "incident_type": payload.get("incident_type", "Pothole"),
        "severity": payload.get("severity", "Medium"),
        "reported_by": payload.get("reported_by", "Citizen"),
        "injuries": payload.get("injuries", 0),
        "hour": timestamp.hour,
        "weekday": timestamp.dayofweek,
        "month": timestamp.month,
        "near_school": float(bool(row["near_school"])),
        "near_hospital": float(bool(row["near_hospital"])),
    })
    return pd.DataFrame([row])


def _metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {"mse": round(float(mean_squared_error(actual, predicted)), 4), "mae": round(float(mean_absolute_error(actual, predicted)), 4), "rmse": round(float(mean_squared_error(actual, predicted) ** 0.5), 4), "r2": round(float(r2_score(actual, predicted)), 4)}


def _cleaning_report(data: dict, frame: pd.DataFrame) -> dict:
    incidents = data["incidents"]
    durations = pd.to_numeric(incidents["resolution_hours"], errors="coerce")
    timestamp_valid = incidents["timestamp"].notna()
    resolved = incidents["resolved"]
    positive_duration = durations.gt(0) & durations.notna()
    eligible = resolved & positive_duration & timestamp_valid
    target = durations.loc[eligible]
    q1, q3 = (float(target.quantile(value)) for value in (.25, .75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    dataset_summary = {}
    for name, table in data.items():
        id_column = next((column for column in table if column.endswith("_id") and column != "location_id"), None)
        dataset_summary[name] = {
            "records": len(table), "columns": list(table.columns),
            "dtypes": {column: str(value) for column, value in table.dtypes.items()},
            "missing_values": {column: int(value) for column, value in table.isna().sum().items()},
            "duplicate_record_ids": int(table[id_column].duplicated().sum()) if id_column else 0,
            "location_ids": int(table["location_id"].nunique()),
        }
    excluded = {
        "unresolved": int((~resolved).sum()),
        "resolved_invalid_or_non_positive_duration": int((resolved & ~positive_duration).sum()),
        "resolved_invalid_timestamp": int((resolved & positive_duration & ~timestamp_valid).sum()),
    }
    return {
        "datasets": dataset_summary,
        "join_validation": {"key": "location_id", "location_rows": len(data["locations"]), "unique_location_keys": int(data["locations"]["location_id"].nunique()), "incident_rows_before_join": len(incidents), "incident_rows_after_many_to_one_join": len(data["incidents"].merge(data["locations"], on="location_id", how="left", validate="many_to_one")), "all_incident_location_ids_known": bool(set(incidents["location_id"]).issubset(set(data["locations"]["location_id"])))},
        "incident_resolution_training": {"original_records": len(incidents), "resolved_records": int(resolved.sum()), "excluded_records": int(len(incidents) - eligible.sum()), "excluded_by_reason": excluded, "usable_records": len(frame), "target_outlier_check": {"method": "1.5 IQR fences; flagged, not dropped", "lower_fence": round(lower, 3), "upper_fence": round(upper, 3), "flagged_records": int(((target < lower) | (target > upper)).sum())}},
        "note": "No valid positive resolved record was dropped as an outlier. Feature imputers, category vocabularies and scalers are fitted using the chronological training partition only.",
    }


def train_and_evaluate(epochs: int = 600, learning_rate: float = 0.03, progress_callback: Callable[[dict], None] | None = None) -> dict:
    started = time.perf_counter()
    data = load_data()
    frame = build_training_frame()
    if len(frame) < 30:
        raise ValueError("At least 30 resolved incidents with valid durations are required")
    targets = frame["resolution_hours"].to_numpy(dtype=float)
    count = len(frame)
    test_start = max(1, int(count * 0.8))
    development = frame.iloc[:test_start]
    test = frame.iloc[test_start:]
    train_end = max(1, int(len(development) * 0.8))
    train = development.iloc[:train_end]
    validation = development.iloc[train_end:]
    if len(train) < 2 or validation.empty or test.empty:
        raise ValueError("Chronological training requires non-empty train, validation, and 20% test partitions")
    model = LinearRegressionScratch(learning_rate=learning_rate, epochs=epochs, patience=100)
    epoch_lines: list[str] = []

    def record_epoch(progress: dict[str, float | int]) -> None:
        line = f"Epoch {progress['epoch']}/{epochs} | Training MSE: {progress['training_mse']:.6f} | Validation MSE: {progress['validation_mse']:.6f}"
        epoch_lines.append(line)
        print(line, flush=True)
        if progress_callback is not None:
            progress_callback(progress)

    model.fit(train, train["resolution_hours"].to_numpy(), validation, validation["resolution_hours"].to_numpy(), CATEGORICAL, NUMERIC, on_epoch=record_epoch)
    predictions = np.maximum(0, model.predict(test))
    baseline = np.full(len(test), float(train["resolution_hours"].median()))

    transformer = ColumnTransformer([
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL),
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
    ])
    forest = Pipeline([("preprocess", transformer), ("model", RandomForestRegressor(n_estimators=160, min_samples_leaf=3, random_state=42, n_jobs=-1))])
    forest.fit(train[CATEGORICAL + NUMERIC], train["resolution_hours"])
    forest_predictions = np.maximum(0, forest.predict(test[CATEGORICAL + NUMERIC]))
    forest_temporary = FOREST_PATH.with_suffix(FOREST_PATH.suffix + ".tmp")
    joblib.dump(forest, forest_temporary)
    forest_temporary.replace(FOREST_PATH)
    scratch_metrics = _metrics(test["resolution_hours"].to_numpy(), predictions)
    baseline_metrics = _metrics(test["resolution_hours"].to_numpy(), baseline)
    forest_metrics = _metrics(test["resolution_hours"].to_numpy(), forest_predictions)
    candidates = {
        "NumPy Linear Regression": scratch_metrics["mae"],
        "Training median baseline": baseline_metrics["mae"],
        "Random Forest comparison": forest_metrics["mae"],
    }
    selected_predictor = min(candidates, key=candidates.get)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    elapsed = time.perf_counter() - started
    actual_training_mse = float(np.mean((model.predict(train) - train["resolution_hours"].to_numpy()) ** 2))
    actual_validation_mse = float(np.mean((model.predict(validation) - validation["resolution_hours"].to_numpy()) ** 2))
    best_epoch = int(np.argmin(model.validation_loss)) + 1
    cleaning_report = _cleaning_report(data, frame)
    CLEANING_REPORT_PATH.write_text(json.dumps(cleaning_report, indent=2), encoding="utf-8")
    TRAINING_LOG_PATH.write_text("\n".join(epoch_lines) + "\n", encoding="utf-8")
    stopped_early = len(model.training_loss) < epochs
    report = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_duration_seconds": round(elapsed, 3),
        "target": "resolution_hours",
        "algorithm": "NumPy full-batch gradient descent; MSE objective; standardized numeric features and target; one-hot categorical encoding; explicit intercept",
        "data_split": {"method": "chronological 80/20 test holdout; last 20% by timestamp reserved for test; final 20% of the first 80% used only for validation", "development_records": len(development), "train_records": len(train), "validation_records": len(validation), "test_records": len(test), "excluded_records": len(data["incidents"]) - len(frame), "excluded_reason": cleaning_report["incident_resolution_training"]["excluded_by_reason"]},
        "data_cleaning_report": str(CLEANING_REPORT_PATH.relative_to(ROOT)),
        "training_log": str(TRAINING_LOG_PATH.relative_to(ROOT)),
        "artifact": str(MODEL_PATH.relative_to(ROOT)),
        "manual_linear_regression": {"metrics": _metrics(test["resolution_hours"].to_numpy(), predictions), "training_loss": model.training_loss, "validation_loss": model.validation_loss, "epochs_run": len(model.training_loss), "epochs_requested": epochs, "best_epoch": best_epoch, "best_validation_mse": round(float(model.validation_loss[best_epoch - 1]), 4), "learning_rate": model.learning_rate, "final_training_mse": round(actual_training_mse, 4), "final_validation_mse": round(actual_validation_mse, 4), "stopped_early": stopped_early, "converged": stopped_early, "feature_names": model.encoder.feature_names if model.encoder else []},
        "training_median_baseline": {"prediction": round(float(train["resolution_hours"].median()), 4), "metrics": baseline_metrics},
        "random_forest_comparison": {"metrics": forest_metrics, "note": "Scikit-learn comparison only; the primary model above is implemented manually."},
        "selected_production_predictor": {"name": selected_predictor, "selection_metric": "MAE on the shared chronological test set", "mae_by_candidate": candidates, "reason": "Selected by lower test-set MAE; the scratch model remains available for evaluation and comparison."},
        "limitations": ["The evaluation is a chronological holdout from one synthetic year.", "Locations can recur across time partitions; temporal leakage is avoided, but performance on entirely unseen locations is not measured.", "Predictions estimate historical duration patterns and are not guarantees."],
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = train_and_evaluate()
    print(json.dumps({"report": str(REPORT_PATH), "model": str(MODEL_PATH), "metrics": result["manual_linear_regression"]["metrics"], "baseline": result["training_median_baseline"]["metrics"], "epochs": result["manual_linear_regression"]["epochs_run"]}, indent=2))