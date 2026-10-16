from __future__ import annotations

import sqlite3
import threading
import os
from datetime import timedelta
from pathlib import Path
import json
import joblib

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field
from typing import Literal

from .data import ROOT, data_quality, load_data
from . import ml
from ..ml.linear_regression import LinearRegressionScratch
from ..ml.train_linear_regression import CATEGORICAL, NUMERIC, FOREST_PATH, MODEL_PATH as LINEAR_MODEL_PATH, REPORT_PATH as LINEAR_REPORT_PATH, build_prediction_frame, train_and_evaluate


app = FastAPI(title="CivicLens AI", version="1.0.0", description="Synthetic educational civic-operations decision-support prototype.")
FRONTEND_ORIGIN = os.getenv("CIVICLENS_FRONTEND_ORIGIN", "http://127.0.0.1:5173")
app.add_middleware(CORSMiddleware, allow_origins=list({FRONTEND_ORIGIN, "http://localhost:5173", "http://127.0.0.1:5173"}), allow_methods=["*"], allow_headers=["*"])
DB_PATH = ROOT / "civiclens-local.sqlite3"
TRAINING_LOCK = threading.Lock()
TRAINING_STATUS: dict = {"status": "idle", "error": None}
TRAINING_HISTORY: list[dict] = []
LINEAR_MODEL: LinearRegressionScratch | None = None
FOREST_MODEL = None
RF_PIPELINE = None
RF_METADATA: dict | None = None
RF_MODEL_PATH = ROOT / "backend" / "models" / "random_forest_pipeline.joblib"
RF_METADATA_PATH = ROOT / "backend" / "models" / "model_metadata.json"
OPEN_COMPLAINT_STATUSES = {"open", "in progress", "assigned"}
SEVERITY_VALUE = {"Low": 1, "Medium": 2, "High": 3, "Critical": 4}


class PredictionInput(BaseModel):
    location_id: str
    incident_type: str = "Pothole"
    severity: str = "Medium"
    injuries: int = Field(default=0, ge=0)
    reported_by: str = "Citizen"
    category: str = "Road damage"
    priority_reported: str = "Normal"
    timestamp: str | None = None
    created_at: str | None = None


class RandomForestPredictionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    location_id: str = Field(min_length=1, max_length=40)
    incident_type: Literal["Pothole", "Signal failure", "Streetlight outage", "Blocked drain", "Collision", "Flooding"]
    severity: Literal["Low", "Medium", "High", "Critical"]
    reported_by: Literal["Citizen", "Authority", "Police", "Sensor"]
    reported_injuries: int = Field(default=0, ge=0, le=1000)


class NoteInput(BaseModel):
    location_id: str
    note: str = Field(min_length=1, max_length=4000)


def _db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, location_id TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    return connection


def _frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = load_data()
    return data["locations"], data["incidents"], data["complaints"]


def _priority_rows() -> list[dict]:
    locations, incidents, complaints = _frames()
    incident_groups = incidents.groupby("location_id")
    i = incident_groups.agg(incident_count=("incident_id", "count"), injuries=("injuries", "sum"), unresolved_incidents=("resolved", lambda values: int((~values).sum()))).reset_index()
    resolved_incident_time = incidents.loc[incidents["resolved"]].groupby("location_id")["resolution_hours"].mean().rename("average_resolution_hours").reset_index()
    incident_severity = incident_groups["severity"].agg(lambda values: values.value_counts().to_dict()).rename("severity_distribution").reset_index()
    incident_types = incident_groups["incident_type"].agg(lambda values: values.value_counts().to_dict()).rename("incident_type_distribution").reset_index()
    severe = incidents.assign(severe=incidents["severity"].isin(["High", "Critical"])).groupby("location_id")["severe"].sum().rename("severe_incidents").reset_index()
    severity = incidents.assign(severity_points=incidents["severity"].map(SEVERITY_VALUE).fillna(0)).groupby("location_id")["severity_points"].mean().rename("severity_mean").reset_index()
    completed_complaints = complaints.loc[complaints["status"].str.lower().isin({"resolved", "closed"})]
    complaint_days = completed_complaints.groupby("location_id")["days_to_resolution"].mean().rename("complaint_days").reset_index()
    as_of = complaints["created_at"].max()
    active = complaints.loc[complaints["status"].str.lower().isin(OPEN_COMPLAINT_STATUSES)].copy()
    active["age_days"] = (as_of - active["created_at"]).dt.total_seconds() / 86400
    open_ages = active.groupby("location_id")["age_days"].mean().rename("complaint_age_days").reset_index()
    complaints_open = complaints.assign(is_open=complaints["status"].str.lower().isin(OPEN_COMPLAINT_STATUSES)).groupby("location_id").agg(complaint_count=("complaint_id", "count"), open_complaints=("is_open", "sum")).reset_index()
    complaint_categories = complaints.groupby("location_id")["category"].agg(lambda values: values.value_counts().to_dict()).rename("complaint_distribution").reset_index()
    complaint_statuses = complaints.groupby("location_id")["status"].agg(lambda values: values.value_counts().to_dict()).rename("status_distribution").reset_index()
    result = locations.merge(i, on="location_id", how="left", validate="one_to_one").merge(resolved_incident_time, on="location_id", how="left", validate="one_to_one").merge(incident_severity, on="location_id", how="left", validate="one_to_one").merge(incident_types, on="location_id", how="left", validate="one_to_one").merge(severe, on="location_id", how="left", validate="one_to_one").merge(severity, on="location_id", how="left", validate="one_to_one").merge(complaints_open, on="location_id", how="left", validate="one_to_one").merge(complaint_days, on="location_id", how="left", validate="one_to_one").merge(open_ages, on="location_id", how="left", validate="one_to_one").merge(complaint_categories, on="location_id", how="left", validate="one_to_one").merge(complaint_statuses, on="location_id", how="left", validate="one_to_one")
    numeric = ["incident_count", "injuries", "unresolved_incidents", "average_resolution_hours", "severe_incidents", "severity_mean", "complaint_count", "open_complaints", "complaint_days", "complaint_age_days"]
    result[numeric] = result[numeric].fillna(0)
    raw_components = {
        "incident_volume": result["incident_count"].rank(pct=True),
        "severity": result["severity_mean"].rank(pct=True),
        "injury_burden": result["injuries"].rank(pct=True),
        "unresolved_incidents": result["unresolved_incidents"].rank(pct=True),
        "open_complaints": result["open_complaints"].rank(pct=True),
        "road_condition": result["road_condition"].map({"Good": 0, "Fair": 0.33, "Poor": 0.67, "Critical": 1}).fillna(0.33),
        "resolution_delay": result["average_resolution_hours"].rank(pct=True),
        "context": (result["pedestrian_activity"].map({"Low": 0, "Medium": 0.5, "High": 1}).fillna(0) * 0.5 + result["near_school"].astype(float) * 0.25 + result["near_hospital"].astype(float) * 0.25),
    }
    weights = {"incident_volume": .22, "severity": .16, "injury_burden": .12, "unresolved_incidents": .12, "open_complaints": .12, "road_condition": .12, "resolution_delay": .08, "context": .06}
    scores = sum(raw_components[key] * weight for key, weight in weights.items()) * 100
    result["priority_score"] = scores.round(1)
    rows = []
    for idx, row in result.iterrows():
        components = {key: round(float(raw_components[key].iloc[idx] * 100), 1) for key in raw_components}
        independent_signals = sum([row["incident_count"] >= 10, row["severe_incidents"] >= 2, row["injuries"] >= 2, row["unresolved_incidents"] >= 3, row["open_complaints"] >= 2, row["road_condition"] in {"Poor", "Critical"}])
        evidence = "Strong" if independent_signals >= 3 and row["incident_count"] >= 8 else "Moderate" if independent_signals >= 2 or row["incident_count"] >= 8 else "Limited"
        drivers = sorted(((key, value * weights[key]) for key, value in components.items()), key=lambda item: item[1], reverse=True)
        lead_driver = drivers[0][0].replace("_", " ")
        if row["road_condition"] in {"Poor", "Critical"} and row["incident_count"] >= 8:
            recommendation = "Inspect road condition and reconcile incident locations; condition and incident history are both elevated."
        elif row["unresolved_incidents"] >= 3 or row["open_complaints"] >= 2:
            recommendation = "Review the open work queue, ageing, and assigned owner before scheduling a site investigation."
        elif row["severe_incidents"] >= 2 or row["injuries"] >= 2:
            recommendation = "Review the severe-incident records and verify reported injury details with the responsible team."
        else:
            recommendation = f"Validate the {lead_driver} signal against source records and conduct a proportionate field review."
        rows.append({**{key: (value.item() if isinstance(value, np.generic) else value) for key, value in row.to_dict().items()}, "components": components, "evidence_strength": evidence, "recommendation": recommendation, "score_disclaimer": "Screening heuristic, not a probability of harm."})
    rows.sort(key=lambda item: item["priority_score"], reverse=True)
    for rank, row in enumerate(rows, 1):
        row["rank"] = rank
    return rows


def _trend(frame: pd.DataFrame, date_col: str, period: str, group_col: str | None = None) -> list[dict]:
    dates = pd.to_datetime(frame[date_col], errors="coerce")
    freq = {"day": "D", "week": "W-MON", "month": "M"}.get(period)
    if not freq:
        raise HTTPException(status_code=422, detail="period must be day, week, or month")
    work = frame.copy()
    work["period"] = dates.dt.to_period(freq).dt.start_time
    groupers = ["period"] + ([group_col] if group_col else [])
    counts = work.groupby(groupers, dropna=False).size().reset_index(name="count").sort_values("period")
    result = []
    for _, row in counts.iterrows():
        item = {"date": row["period"].strftime("%Y-%m-%d"), "count": int(row["count"])}
        if group_col:
            item[group_col] = str(row[group_col])
        result.append(item)
    return result


@app.on_event("startup")
def startup() -> None:
    global LINEAR_MODEL, FOREST_MODEL, RF_PIPELINE, RF_METADATA, TRAINING_STATUS, TRAINING_HISTORY
    ml.ensure_models()
    if LINEAR_MODEL_PATH.exists():
        try:
            LINEAR_MODEL = LinearRegressionScratch.load(LINEAR_MODEL_PATH)
        except (OSError, ValueError, KeyError, TypeError):
            LINEAR_MODEL = None
    if FOREST_PATH.exists():
        try:
            FOREST_MODEL = joblib.load(FOREST_PATH)
        except (OSError, ValueError, EOFError):
            FOREST_MODEL = None
    if RF_MODEL_PATH.exists() and RF_METADATA_PATH.exists():
        try:
            metadata = json.loads(RF_METADATA_PATH.read_text(encoding="utf-8"))
            pipeline = joblib.load(RF_MODEL_PATH)
            if metadata.get("model_name") != "RandomForestRegressor" or metadata.get("target") != "resolution_hours":
                raise ValueError("RF model metadata does not match the supported target")
            RF_PIPELINE, RF_METADATA = pipeline, metadata
        except (OSError, ValueError, KeyError, EOFError, TypeError):
            RF_PIPELINE, RF_METADATA = None, None
    if LINEAR_REPORT_PATH.exists():
        try:
            report = json.loads(LINEAR_REPORT_PATH.read_text(encoding="utf-8"))
            model_report = report["manual_linear_regression"]
            TRAINING_HISTORY = [{"epoch": index + 1, "training_mse": train_loss, "validation_mse": validation_loss} for index, (train_loss, validation_loss) in enumerate(zip(model_report["training_loss"], model_report["validation_loss"]))]
            TRAINING_STATUS = {"status": "complete" if LINEAR_MODEL is not None else "unavailable", "trained_at": report.get("trained_at"), "epochs_completed": model_report.get("epochs_run"), "train_records": report.get("data_split", {}).get("train_records"), "validation_records": report.get("data_split", {}).get("validation_records"), "test_records": report.get("data_split", {}).get("test_records"), "metrics": model_report.get("metrics"), "artifact_available": LINEAR_MODEL_PATH.exists(), "error": None}
        except (OSError, ValueError, KeyError, TypeError):
            TRAINING_STATUS = {"status": "unavailable", "error": "Saved model report is invalid."}
    with _db() as connection:
        connection.commit()


@app.get("/api/health")
@app.get("/health")
def health() -> dict:
    available = RF_PIPELINE is not None
    return {"status": "ok", "product": "CivicLens AI", "dataset_date_range": data_quality()["incident_date_range"], "model_available": available, "model": "RandomForestRegressor" if available else None, "synthetic": True}


@app.get("/model/metrics")
def random_forest_metrics() -> dict:
    if RF_METADATA is None or RF_PIPELINE is None:
        raise HTTPException(status_code=503, detail="Random Forest model is unavailable. Train it with python -m backend.train_model.")
    return {
        "model_name": RF_METADATA["model_name"],
        "training_dataset_size": RF_METADATA["dataset"]["eligible_resolved_records"],
        "split": RF_METADATA["split"],
        "cross_validation": RF_METADATA["cross_validation"],
        "held_out_test": RF_METADATA["held_out_test"],
        "training_timestamp": RF_METADATA["trained_at"],
        "features": RF_METADATA["feature_schema"],
        "feature_importance": RF_METADATA.get("feature_importance"),
        "target_distribution": RF_METADATA.get("target_distribution"),
        "prediction_distribution": RF_METADATA.get("prediction_distribution"),
        "comparison": {"random_forest_beats_median_on_mae": RF_METADATA["held_out_test"]["random_forest"]["mae"] < RF_METADATA["held_out_test"]["median_baseline"]["mae"]},
    }


def _random_forest_features(payload: RandomForestPredictionInput) -> pd.DataFrame:
    feature_frame = build_prediction_frame({
        "location_id": payload.location_id,
        "incident_type": payload.incident_type,
        "severity": payload.severity,
        "reported_by": payload.reported_by,
        "injuries": payload.reported_injuries,
    })
    required = RF_METADATA["feature_schema"]["input_columns"]
    missing = [column for column in required if column not in feature_frame]
    if missing:
        raise ValueError(f"Missing prediction features: {', '.join(missing)}")
    return feature_frame[required]


def _operational_priority(payload: RandomForestPredictionInput) -> dict:
    critical = payload.severity == "Critical"
    high_injury_collision = payload.incident_type == "Collision" and payload.reported_injuries >= 5
    high_severity_injury = payload.severity == "High" and payload.reported_injuries > 0
    if critical or high_injury_collision or high_severity_injury:
        return {"level": "urgent_human_assessment", "reason": "Critical severity or a high-injury collision/high-severity incident requires immediate human triage under local protocol.", "uses_predicted_duration": False}
    if payload.severity == "High" or payload.reported_injuries > 0:
        return {"level": "priority_human_review", "reason": "High severity or reported injuries warrant prompt human review.", "uses_predicted_duration": False}
    return {"level": "standard_human_review", "reason": "Review according to the normal operations workflow; the model does not determine safety urgency.", "uses_predicted_duration": False}


def _prediction_support_warnings(payload: RandomForestPredictionInput) -> list[str]:
    support = (RF_METADATA or {}).get("feature_schema", {}).get("training_numeric_support", {}).get("injuries")
    if support and (payload.reported_injuries < support["min"] or payload.reported_injuries > support["max"]):
        return [f"reported_injuries={payload.reported_injuries} is outside the training range [{support['min']}, {support['max']}]; estimate is extrapolative and especially uncertain."]
    return []


@app.post("/predict")
def predict_random_forest(payload: RandomForestPredictionInput) -> dict:
    if RF_PIPELINE is None or RF_METADATA is None:
        raise HTTPException(status_code=503, detail="Random Forest model is unavailable. Train it with python -m backend.train_model.")
    try:
        features = _random_forest_features(payload)
        prediction = float(RF_PIPELINE.predict(features)[0])
        if not np.isfinite(prediction):
            raise ValueError("Model returned a non-finite estimate")
        return {
            "estimated_resolution_hours": round(max(0.0, prediction), 2),
            "model": RF_METADATA["model_name"],
            "unit": "hours",
            "disclaimer": "Historical model estimate, not a guarantee.",
            "limitations": ["This predicts resolution duration, not first-response time.", "One synthetic year provides limited evidence for generalization.", "Resolution estimates must not determine emergency or safety priority."],
            "input_support_warnings": _prediction_support_warnings(payload),
            "priority_recommendation": _operational_priority(payload),
        }
    except ValueError as exc:
        message = str(exc)
        if message.startswith("Unknown location_id:"):
            raise HTTPException(status_code=422, detail=message) from exc
        raise HTTPException(status_code=422, detail=f"Prediction features could not be processed: {message}") from exc


@app.get("/api/dashboard")
def dashboard() -> dict:
    locations, incidents, complaints = _frames()
    priorities = _priority_rows()
    open_incidents = int((~incidents["resolved"]).sum())
    open_complaints = int(complaints["status"].str.lower().isin(OPEN_COMPLAINT_STATUSES).sum())
    recent_cut = incidents["timestamp"].max() - timedelta(days=30)
    previous_cut = recent_cut - timedelta(days=30)
    recent = int((incidents["timestamp"] >= recent_cut).sum())
    previous = int(((incidents["timestamp"] >= previous_cut) & (incidents["timestamp"] < recent_cut)).sum())
    return {
        "locations": len(locations), "incidents": len(incidents), "complaints": len(complaints),
        "open_incidents": open_incidents, "open_complaints": open_complaints,
        "average_resolution_hours": round(float(incidents.loc[incidents["resolved"], "resolution_hours"].mean()), 1),
        "high_priority_locations": sum(row["priority_score"] >= 75 for row in priorities),
        "repeated_incident_locations": int((incidents.groupby("location_id").size() >= 3).sum()),
        "incident_date_range": data_quality()["incident_date_range"],
        "recent_trend": {"last_30_days": recent, "prior_30_days": previous, "change_percent": round((recent - previous) / max(previous, 1) * 100, 1)},
        "priority_preview": priorities[:5], "synthetic": True,
    }


@app.get("/api/locations")
def locations_list(zone: str | None = None, road_condition: str | None = None, incident_type: str | None = None, severity: str | None = None, complaint_category: str | None = None, status: str | None = None, search: str | None = None, page: int = Query(1, ge=1), page_size: int = Query(100, ge=1, le=1000)) -> dict:
    locations, incidents, complaints = _frames()
    frame = pd.DataFrame(_priority_rows())
    if zone: frame = frame[frame["zone"] == zone]
    if road_condition: frame = frame[frame["road_condition"] == road_condition]
    if search: frame = frame[frame["location_id"].str.contains(search, case=False, regex=False)]
    if incident_type:
        ids = set(incidents.loc[incidents["incident_type"] == incident_type, "location_id"])
        frame = frame[frame["location_id"].isin(ids)]
    if severity:
        ids = set(incidents.loc[incidents["severity"] == severity, "location_id"])
        frame = frame[frame["location_id"].isin(ids)]
    if complaint_category:
        ids = set(complaints.loc[complaints["category"] == complaint_category, "location_id"])
        frame = frame[frame["location_id"].isin(ids)]
    if status:
        if status.lower() in OPEN_COMPLAINT_STATUSES:
            ids = set(complaints.loc[complaints["status"].str.lower() == status.lower(), "location_id"])
            frame = frame[frame["location_id"].isin(ids)]
        elif status.lower() in {"unresolved", "open"}:
            ids = set(incidents.loc[~incidents["resolved"], "location_id"]) | set(complaints.loc[complaints["status"].str.lower().isin(OPEN_COMPLAINT_STATUSES), "location_id"])
            frame = frame[frame["location_id"].isin(ids)]
    total = len(frame)
    start = (page - 1) * page_size
    return {"items": frame.iloc[start:start + page_size].to_dict(orient="records"), "total": total, "page": page, "page_size": page_size, "filters": {"zones": sorted(locations["zone"].dropna().unique().tolist()), "road_conditions": sorted(locations["road_condition"].dropna().unique().tolist()), "incident_types": sorted(incidents["incident_type"].dropna().unique().tolist()), "severities": sorted(incidents["severity"].dropna().unique().tolist()), "complaint_categories": sorted(complaints["category"].dropna().unique().tolist()), "statuses": sorted(complaints["status"].dropna().unique().tolist())}}


@app.get("/api/locations/{location_id}")
def location_detail(location_id: str) -> dict:
    locations, incidents, complaints = _frames()
    match = locations.loc[locations["location_id"] == location_id]
    if match.empty: raise HTTPException(status_code=404, detail=f"Unknown location_id: {location_id}")
    location = match.iloc[0].to_dict()
    location["near_school"] = bool(location["near_school"])
    location["near_hospital"] = bool(location["near_hospital"])
    i = incidents.loc[incidents["location_id"] == location_id].sort_values("timestamp", ascending=False)
    c = complaints.loc[complaints["location_id"] == location_id].sort_values("created_at", ascending=False)
    priority = next(row for row in _priority_rows() if row["location_id"] == location_id)
    same_zone = incidents.merge(locations[["location_id", "zone"]], on="location_id", validate="many_to_one").loc[lambda rows: rows["zone"] == location["zone"]]
    manual_prediction = float(LINEAR_MODEL.predict(build_prediction_frame({"location_id": location_id}))[0]) if LINEAR_MODEL is not None else None
    return {"location": location, "priority": priority, "incidents": i.assign(timestamp=i["timestamp"].astype(str)).to_dict(orient="records"), "complaints": c.assign(created_at=c["created_at"].astype(str)).to_dict(orient="records"), "severity_distribution": i["severity"].value_counts().to_dict(), "complaint_distribution": c["category"].value_counts().to_dict(), "resolution_hours": {"mean": round(float(i.loc[i["resolved"], "resolution_hours"].mean()), 1) if i["resolved"].any() else None, "median": round(float(i.loc[i["resolved"], "resolution_hours"].median()), 1) if i["resolved"].any() else None}, "comparison": {"zone": location["zone"], "zone_incident_count": len(same_zone), "zone_average_per_location": round(len(same_zone) / max(1, same_zone["location_id"].nunique()), 2), "location_incident_count": len(i)}, "prediction": {"incident_resolution_hours": manual_prediction, "incident_resolution_comparison_hours": ml.predict_resolution("incident", {"location_id": location_id}), "complaint_resolution_days": ml.predict_resolution("complaint", {"location_id": location_id})}, "limitations": ["Synthetic coordinates are not real streets or civic reports.", "One year of history cannot establish long-term seasonality or causality.", "Predictions are estimates from historical metadata, not guarantees."]}


@app.get("/api/priorities")
def priorities(limit: int = Query(1000, ge=1, le=1000), sort: str = "priority_score", direction: str = "desc") -> dict:
    rows = _priority_rows()
    allowed = {"priority_score", "incident_count", "severe_incidents", "injuries", "complaint_count", "open_complaints", "average_resolution_hours", "location_id"}
    if sort not in allowed: raise HTTPException(status_code=422, detail=f"sort must be one of: {', '.join(sorted(allowed))}")
    rows.sort(key=lambda item: item[sort], reverse=direction.lower() != "asc")
    for rank, row in enumerate(rows, 1): row["rank"] = rank
    return {"items": rows[:limit], "total": len(rows), "formula": {"weights": {"incident_volume": .22, "severity": .16, "injury_burden": .12, "unresolved_incidents": .12, "open_complaints": .12, "road_condition": .12, "resolution_delay": .08, "context": .06}, "normalization": "Per-component percentile ranks within this dataset; road/context mapped to bounded 0-1 scales.", "evidence_rules": "Strong: at least three independent indicators and 8+ incidents; Moderate: at least two indicators or 8+ incidents; otherwise Limited."}}


@app.get("/api/trends/incidents")
@app.get("/api/incidents/trends")
def incident_trends(period: str = "week") -> dict:
    _, incidents, _ = _frames()
    return {"items": _trend(incidents, "timestamp", period), "period": period, "date_range": data_quality()["incident_date_range"]}


@app.get("/api/trends/complaints")
@app.get("/api/complaints/trends")
def complaint_trends(period: str = "week", group_by: str = "category") -> dict:
    _, _, complaints = _frames()
    if group_by not in {"category", "status", "priority_reported"}: raise HTTPException(status_code=422, detail="group_by must be category, status, or priority_reported")
    return {"items": _trend(complaints, "created_at", period, group_by), "period": period, "group_by": group_by, "date_range": data_quality()["complaint_date_range"]}


@app.get("/api/forecast")
def forecast() -> dict:
    _, incidents, _ = _frames()
    weekly = incidents.set_index("timestamp").resample("W-MON").size().asfreq("W-MON", fill_value=0).astype(float)
    test_size = min(8, max(0, len(weekly) - 8))
    errors_last, errors_avg = [], []
    if test_size:
        start = len(weekly) - test_size
        for idx in range(start, len(weekly)):
            prior = weekly.iloc[max(0, idx - 4):idx]
            if len(prior):
                errors_avg.append(abs(float(weekly.iloc[idx]) - float(prior.mean())))
                errors_last.append(abs(float(weekly.iloc[idx]) - float(prior.iloc[-1])))
    history = [{"date": date.strftime("%Y-%m-%d"), "count": int(value)} for date, value in weekly.items()]
    horizon = 4
    estimate = float(weekly.tail(4).mean()) if len(weekly) else 0
    forecast_rows = []
    for offset in range(1, horizon + 1):
        date = (weekly.index[-1] + timedelta(weeks=offset)).strftime("%Y-%m-%d")
        spread = 1.96 * np.sqrt(max(estimate, 1))
        forecast_rows.append({"date": date, "predicted": round(estimate, 1), "lower": round(max(0, estimate - spread), 1), "upper": round(estimate + spread, 1)})
    return {"history": history, "forecast": forecast_rows, "baseline": "Mean weekly incident count across the latest four observed weeks; Poisson-style interval is indicative only.", "evaluation": {"test_weeks": test_size, "rolling_four_week_mae": round(float(np.mean(errors_avg)), 2) if errors_avg else None, "previous_week_mae": round(float(np.mean(errors_last)), 2) if errors_last else None}, "limitations": "One synthetic year supports only a short baseline outlook; no stable seasonal pattern is established."}


@app.get("/api/alerts")
def alerts() -> dict:
    locations, incidents, complaints = _frames()
    ranked = _priority_rows()
    result = []
    for row in ranked:
        if row["incident_count"] >= 16:
            result.append({"id": f"repeat-{row['location_id']}", "type": "Repeated incidents", "location_id": row["location_id"], "severity": "review", "reason": f"{row['incident_count']} incidents recorded, above the 15-record review threshold.", "evidence": {"incident_count": row["incident_count"], "threshold": 15}})
        if row["open_complaints"] >= 3:
            result.append({"id": f"open-{row['location_id']}", "type": "Multiple open complaints", "location_id": row["location_id"], "severity": "review", "reason": f"{row['open_complaints']} complaints are open, in progress, or assigned.", "evidence": {"open_complaints": row["open_complaints"], "statuses": sorted(OPEN_COMPLAINT_STATUSES)}})
        if row["road_condition"] in {"Poor", "Critical"} and row["incident_count"] >= 10:
            result.append({"id": f"condition-{row['location_id']}", "type": "Condition and history", "location_id": row["location_id"], "severity": "review", "reason": f"Road condition is {row['road_condition'].lower()} with {row['incident_count']} incidents; the combination warrants verification, not a danger conclusion.", "evidence": {"road_condition": row["road_condition"], "incident_count": row["incident_count"]}})
        if row["average_resolution_hours"] >= 72:
            result.append({"id": f"delay-{row['location_id']}", "type": "Long resolution times", "location_id": row["location_id"], "severity": "review", "reason": f"Mean recorded resolution time is {row['average_resolution_hours']:.1f} hours (review threshold: 72).", "evidence": {"mean_hours": round(row["average_resolution_hours"], 1), "threshold_hours": 72}})
        if row["near_school"] or row["near_hospital"]:
            result.append({"id": f"context-{row['location_id']}", "type": "Context for human review", "location_id": row["location_id"], "severity": "context", "reason": "A supplied context flag indicates proximity to a school or hospital. This is not evidence of danger and is included only to prompt proportionate human review.", "evidence": {"near_school": bool(row["near_school"]), "near_hospital": bool(row["near_hospital"]), "incident_count": row["incident_count"]}})
    weekly = incidents.assign(week=incidents["timestamp"].dt.to_period("W").dt.start_time).groupby(["location_id", "week"]).size().unstack(fill_value=0)
    if weekly.shape[1] >= 16:
        recent = weekly.iloc[:, -4:].sum(axis=1)
        baseline = weekly.iloc[:, -16:-4].mean(axis=1) * 4
        spread = weekly.iloc[:, -16:-4].std(axis=1).fillna(0) * 2 * 2.5
        for loc_id in recent.index[(recent >= baseline + spread + 3) & (recent >= 6)]:
            result.append({"id": f"anomaly-{loc_id}", "type": "Unusual incident volume", "location_id": loc_id, "severity": "review", "reason": "Recent four-week count exceeds the prior 12-week per-location baseline by more than 2.5 standard deviations and at least three records.", "evidence": {"observed_four_weeks": int(recent[loc_id]), "baseline_four_weeks": round(float(baseline[loc_id]), 1), "threshold_count": round(float(baseline[loc_id] + spread[loc_id] + 3), 1)}})
    result.sort(key=lambda alert: (alert["severity"] != "review", alert["location_id"], alert["type"]))
    return {"items": result, "count": len(result), "note": "Alerts are screening signals from synthetic records, not confirmed safety problems or live reports."}


@app.get("/api/models")
@app.get("/api/models/status")
def model_status() -> dict:
    ml.ensure_models()
    report = json.loads(LINEAR_REPORT_PATH.read_text(encoding="utf-8")) if LINEAR_REPORT_PATH.exists() else None
    selected = report.get("selected_production_predictor") if report else None
    if report and selected is None:
        candidate_metrics = {
            "NumPy Linear Regression": report["manual_linear_regression"]["metrics"],
            "Training median baseline": report["training_median_baseline"]["metrics"],
            "Random Forest comparison": report["random_forest_comparison"]["metrics"],
        }
        selected_name = min(candidate_metrics, key=lambda name: candidate_metrics[name]["mae"])
        selected = {"name": selected_name, "selection_metric": "MAE on the shared chronological test set", "mae_by_candidate": {name: values["mae"] for name, values in candidate_metrics.items()}}
    return {"status": "trained" if report and LINEAR_MODEL is not None else "unavailable", "model_type": "NumPy gradient-descent linear regression", "target": "resolution_hours", "training": TRAINING_STATUS, "trained_at": report.get("trained_at") if report else None, "training_records": report.get("data_split", {}).get("train_records") if report else 0, "validation_records": report.get("data_split", {}).get("validation_records") if report else 0, "test_records": report.get("data_split", {}).get("test_records") if report else 0, "evaluation": report.get("manual_linear_regression", {}).get("metrics") if report else None, "selected_predictor": selected, "artifact_available": LINEAR_MODEL_PATH.exists() and LINEAR_MODEL is not None, "forest_artifact_available": FOREST_PATH.exists() and FOREST_MODEL is not None, "error": TRAINING_STATUS.get("error"), "manual_linear_regression": LINEAR_MODEL is not None, "metrics": ml.METRICS.get("models", {}), "forecast": "Four-week rolling baseline; one-year history limits validation.", "anomaly": "Per-location comparison of latest four weeks with preceding 12 weeks.", "priority": "Transparent percentile-weighted screening heuristic; not validated probability."}


@app.get("/api/models/metrics")
def model_metrics() -> dict:
    report = json.loads(LINEAR_REPORT_PATH.read_text(encoding="utf-8")) if LINEAR_REPORT_PATH.exists() else None
    cleaning_path = ROOT / "reports" / "data_cleaning_report.json"
    cleaning = json.loads(cleaning_path.read_text(encoding="utf-8")) if cleaning_path.exists() else None
    return {"manual_linear_regression": report, "data_cleaning": cleaning, "other_models": ml.METRICS.get("models", {})}


@app.get("/api/models/training-history")
def training_history() -> dict:
    return {"status": TRAINING_STATUS.get("status", "idle"), "items": TRAINING_HISTORY, "count": len(TRAINING_HISTORY)}


def _predict_selected(payload: PredictionInput) -> dict:
    if LINEAR_MODEL is None or not LINEAR_REPORT_PATH.exists():
        raise HTTPException(status_code=503, detail="No valid trained incident-resolution model is available.")
    report = json.loads(LINEAR_REPORT_PATH.read_text(encoding="utf-8"))
    selection = report.get("selected_production_predictor", {}).get("name")
    if selection is None:
        candidates = {
            "NumPy Linear Regression": report["manual_linear_regression"]["metrics"]["mae"],
            "Training median baseline": report["training_median_baseline"]["metrics"]["mae"],
            "Random Forest comparison": report["random_forest_comparison"]["metrics"]["mae"],
        }
        selection = min(candidates, key=candidates.get)
    features = build_prediction_frame(payload.model_dump(exclude_none=True))
    linear_estimate = max(0.0, float(LINEAR_MODEL.predict(features)[0]))
    if selection == "Training median baseline":
        estimate = float(report["training_median_baseline"]["prediction"])
    elif selection == "Random Forest comparison" and FOREST_MODEL is not None:
        estimate = max(0.0, float(FOREST_MODEL.predict(features[CATEGORICAL + NUMERIC])[0]))
    else:
        estimate = linear_estimate
        selection = "NumPy Linear Regression"
    return {"estimate": round(estimate, 2), "scratch_linear_regression_estimate": round(linear_estimate, 2), "unit": "hours", "selected_model": selection, "caveat": "Historical estimate selected by lower test-set MAE; synthetic-data prediction is not a guarantee."}


def _train_manual_model() -> None:
    global TRAINING_STATUS, TRAINING_HISTORY, LINEAR_MODEL
    try:
        TRAINING_HISTORY = []

        def record_epoch(progress: dict) -> None:
            global TRAINING_STATUS
            TRAINING_HISTORY.append(progress)
            TRAINING_STATUS = {**TRAINING_STATUS, "status": "running", "current_epoch": progress["epoch"], "training_mse": progress["training_mse"], "validation_mse": progress["validation_mse"]}

        result = train_and_evaluate(progress_callback=record_epoch)
        LINEAR_MODEL = LinearRegressionScratch.load(LINEAR_MODEL_PATH)
        model_report = result["manual_linear_regression"]
        TRAINING_STATUS = {"status": "complete", "trained_at": result["trained_at"], "epochs_completed": model_report["epochs_run"], "train_records": result["data_split"]["train_records"], "validation_records": result["data_split"]["validation_records"], "test_records": result["data_split"]["test_records"], "metrics": model_report["metrics"], "artifact_available": LINEAR_MODEL_PATH.exists(), "error": None}
    except Exception as exc:
        TRAINING_STATUS = {"status": "failed", "error": str(exc)}
    finally:
        TRAINING_LOCK.release()


@app.post("/api/models/train", status_code=202)
def train_models() -> dict:
    global TRAINING_STATUS
    if not TRAINING_LOCK.acquire(blocking=False):
        return {"status": "running", "message": "A training run is already active."}
    TRAINING_STATUS = {"status": "running", "started_at": pd.Timestamp.now(tz="UTC").isoformat(), "current_epoch": 0, "epochs_requested": 600}
    threading.Thread(target=_train_manual_model, daemon=True).start()
    return {"status": "running", "message": "Manual linear regression training started; poll /api/models/status for the actual result."}


@app.post("/api/models/predict")
def predict_manual_model(payload: PredictionInput) -> dict:
    try:
        return _predict_selected(payload)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/models/retrain")
def retrain() -> dict:
    try:
        load_data.cache_clear()
        return {"status": "trained", "metrics": ml.train_all()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Model training failed: {exc}") from exc


@app.get("/api/data-quality")
def quality() -> dict:
    return data_quality()


@app.post("/api/data/refresh")
def refresh() -> dict:
    load_data.cache_clear()
    try:
        report = data_quality()
        return {"status": "refreshed", "report": report}
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not refresh datasets: {exc}") from exc


@app.post("/api/predictions/resolution")
def resolution_prediction(payload: PredictionInput, kind: str = Query("incident", pattern="^(incident|complaint)$")) -> dict:
    try:
        if kind == "incident" and LINEAR_MODEL is not None:
            return {"kind": kind, **_predict_selected(payload)}
        value = ml.predict_resolution(kind, payload.model_dump(exclude_none=True))
        return {"kind": kind, "estimate": round(value, 2), "unit": "hours" if kind == "incident" else "days", "caveat": "Historical metadata estimate, not a guarantee; evaluation metrics are available in Model Lab."}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/notes/{location_id}")
def get_notes(location_id: str) -> dict:
    with _db() as connection:
        rows = connection.execute("SELECT id, location_id, note, created_at FROM notes WHERE location_id = ? ORDER BY id DESC", (location_id,)).fetchall()
    return {"items": [dict(row) for row in rows]}


@app.post("/api/notes", status_code=201)
def add_note(payload: NoteInput) -> dict:
    with _db() as connection:
        cursor = connection.execute("INSERT INTO notes (location_id, note) VALUES (?, ?)", (payload.location_id, payload.note.strip()))
        connection.commit()
        row = connection.execute("SELECT id, location_id, note, created_at FROM notes WHERE id = ?", (cursor.lastrowid,)).fetchone()
    return dict(row)