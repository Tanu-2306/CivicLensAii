from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .data import ROOT, load_data


MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
METRICS_PATH = MODEL_DIR / "metrics.json"
MODELS: dict[str, object] = {}
METRICS: dict = {}
TRAINED_AT: str | None = None

INCIDENT_CATEGORICAL = ["incident_type", "severity", "reported_by", "zone", "road_type", "road_condition", "pedestrian_activity"]
INCIDENT_NUMERIC = ["injuries", "hour", "weekday", "month", "population_density_per_km2", "near_school", "near_hospital"]
COMPLAINT_CATEGORICAL = ["category", "priority_reported", "zone", "road_type", "road_condition", "pedestrian_activity"]
COMPLAINT_NUMERIC = ["hour", "weekday", "month", "population_density_per_km2", "near_school", "near_hospital"]
CLASSIFICATION_CATEGORICAL = ["priority_reported", "zone", "road_type", "road_condition", "pedestrian_activity"]
CLASSIFICATION_NUMERIC = ["hour", "weekday", "month", "population_density_per_km2", "near_school", "near_hospital"]


def _preprocessor(categorical: list[str], numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer([
        ("categorical", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("numeric", SimpleImputer(strategy="median"), numeric),
    ])


def _features(frame: pd.DataFrame, date_col: str, location: pd.DataFrame) -> pd.DataFrame:
    joined = frame.merge(location, on="location_id", how="left", validate="many_to_one")
    dates = pd.to_datetime(joined[date_col], errors="coerce")
    joined["hour"] = dates.dt.hour
    joined["weekday"] = dates.dt.dayofweek
    joined["month"] = dates.dt.month
    joined["near_school"] = joined["near_school"].astype(float)
    joined["near_hospital"] = joined["near_hospital"].astype(float)
    return joined


def _fit_regression(frame: pd.DataFrame, target: str, categorical: list[str], numeric: list[str], date_col: str, title: str, eligible: pd.Series) -> tuple[Pipeline, dict]:
    valid_target = frame[target].notna() & (frame[target] > 0)
    usable = frame.loc[valid_target & eligible].sort_values(date_col).copy()
    usable = _features(usable, date_col, load_data()["locations"])
    feature_cols = categorical + numeric
    split = max(1, min(len(usable) - 1, int(len(usable) * 0.8)))
    train, test = usable.iloc[:split], usable.iloc[split:]
    candidates = {
        "Median baseline": Pipeline([("preprocess", _preprocessor(categorical, numeric)), ("model", DummyRegressor(strategy="median"))]),
        "Random forest": Pipeline([("preprocess", _preprocessor(categorical, numeric)), ("model", RandomForestRegressor(n_estimators=120, min_samples_leaf=3, random_state=42, n_jobs=-1))]),
    }
    results = {}
    for name, model in candidates.items():
        model.fit(train[feature_cols], train[target])
        predictions = np.maximum(0, model.predict(test[feature_cols]))
        results[name] = {"mae": round(float(mean_absolute_error(test[target], predictions)), 3), "rmse": round(float(mean_squared_error(test[target], predictions) ** 0.5), 3)}
    selected_name = min(results, key=lambda name: results[name]["mae"])
    chosen = candidates[selected_name]
    chosen.fit(usable[feature_cols], usable[target])
    return chosen, {"name": title, "target": target, "training_records": int(len(usable)), "excluded_records": int(len(frame) - len(usable)), "excluded_reason": "Unresolved status or invalid/non-positive target duration", "evaluation": results, "selected_model": selected_name, "split": "chronological 80/20 holdout", "features": feature_cols}


def train_all() -> dict:
    global MODELS, METRICS, TRAINED_AT
    data = load_data()
    locations, incidents, complaints = data["locations"], data["incidents"], data["complaints"]
    incident_source = incidents.copy()
    incident_source["resolution_hours"] = pd.to_numeric(incident_source["resolution_hours"], errors="coerce")
    complaint_source = complaints.copy()
    complaint_source["days_to_resolution"] = pd.to_numeric(complaint_source["days_to_resolution"], errors="coerce")
    incident_model, incident_metrics = _fit_regression(incident_source, "resolution_hours", INCIDENT_CATEGORICAL, INCIDENT_NUMERIC, "timestamp", "Incident resolution estimate", incidents["resolved"])
    complaint_model, complaint_metrics = _fit_regression(complaint_source, "days_to_resolution", COMPLAINT_CATEGORICAL, COMPLAINT_NUMERIC, "created_at", "Complaint resolution estimate", complaints["status"].str.lower().isin({"resolved", "closed"}))

    complaint_features = _features(complaints, "created_at", locations)
    classifier_columns = CLASSIFICATION_CATEGORICAL + CLASSIFICATION_NUMERIC
    ordered = complaint_features.sort_values("created_at")
    cut = int(len(ordered) * 0.8)
    classifier = Pipeline([("preprocess", _preprocessor(CLASSIFICATION_CATEGORICAL, CLASSIFICATION_NUMERIC)), ("model", RandomForestClassifier(n_estimators=160, min_samples_leaf=3, random_state=42, class_weight="balanced", n_jobs=-1))])
    class_metrics: dict = {"name": "Complaint category metadata diagnostic", "target": "category", "features": classifier_columns, "limitation": "Metadata-only; no complaint text is supplied, so this is not language understanding."}
    if ordered["category"].nunique() > 1 and ordered.iloc[:cut]["category"].value_counts().min() >= 2:
        classifier.fit(ordered.iloc[:cut][classifier_columns], ordered.iloc[:cut]["category"])
        predicted = classifier.predict(ordered.iloc[cut:][classifier_columns])
        actual = ordered.iloc[cut:]["category"]
        labels = sorted(ordered["category"].unique())
        class_metrics.update({"training_records": cut, "test_records": len(actual), "split": "chronological 80/20 holdout", "accuracy": round(float(accuracy_score(actual, predicted)), 3), "macro_f1": round(float(f1_score(actual, predicted, labels=labels, average="macro", zero_division=0)), 3), "labels": labels, "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist()})
    else:
        classifier = None
        class_metrics["status"] = "Insufficient classes for evaluation"

    MODELS = {"incident_resolution": incident_model, "complaint_resolution": complaint_model}
    if classifier is not None:
        MODELS["complaint_category"] = classifier
    TRAINED_AT = datetime.now(timezone.utc).isoformat()
    METRICS = {"trained_at": TRAINED_AT, "models": {"incident_resolution": incident_metrics, "complaint_resolution": complaint_metrics, "complaint_category": class_metrics}}
    for name, model in MODELS.items():
        joblib.dump(model, MODEL_DIR / f"{name}.joblib")
    METRICS_PATH.write_text(json.dumps(METRICS, indent=2), encoding="utf-8")
    return METRICS


def ensure_models() -> None:
    global MODELS, METRICS, TRAINED_AT
    if METRICS_PATH.exists() and all((MODEL_DIR / f"{name}.joblib").exists() for name in ("incident_resolution", "complaint_resolution")):
        try:
            MODELS = {name: joblib.load(MODEL_DIR / f"{name}.joblib") for name in ("incident_resolution", "complaint_resolution")}
            classifier_path = MODEL_DIR / "complaint_category.joblib"
            if classifier_path.exists():
                MODELS["complaint_category"] = joblib.load(classifier_path)
            METRICS = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            TRAINED_AT = METRICS.get("trained_at")
            return
        except (OSError, ValueError, KeyError, EOFError):
            MODELS, METRICS = {}, {}
    train_all()


def predict_resolution(kind: str, row: dict) -> float:
    ensure_models()
    model_key, categorical, numeric, date_col = (
        ("incident_resolution", INCIDENT_CATEGORICAL, INCIDENT_NUMERIC, "timestamp")
        if kind == "incident" else
        ("complaint_resolution", COMPLAINT_CATEGORICAL, COMPLAINT_NUMERIC, "created_at")
    )
    data = load_data()
    source = pd.DataFrame([row])
    date_value = row.get(date_col, datetime.now(timezone.utc).isoformat())
    source[date_col] = pd.to_datetime([date_value], errors="coerce")
    if kind == "incident":
        for column, default in {"incident_type": "Pothole", "severity": "Medium", "reported_by": "Citizen", "injuries": 0}.items():
            if column not in source:
                source[column] = default
    else:
        for column, default in {"category": "Road damage", "priority_reported": "Normal"}.items():
            if column not in source:
                source[column] = default
    loc_id = str(row.get("location_id", data["locations"].iloc[0]["location_id"]))
    location = data["locations"].loc[data["locations"]["location_id"] == loc_id]
    if location.empty:
        raise ValueError(f"Unknown location_id: {loc_id}")
    features = _features(source, date_col, location)
    return max(0.0, float(MODELS[model_key].predict(features[categorical + numeric])[0]))