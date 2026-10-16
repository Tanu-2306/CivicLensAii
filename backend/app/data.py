from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "locations": ROOT / "locations.csv",
    "incidents": ROOT / "incidents.csv",
    "complaints": ROOT / "public_complaints.csv",
}
REQUIRED = {
    "locations": {"location_id", "zone", "road_type", "road_condition", "population_density_per_km2", "near_school", "near_hospital", "pedestrian_activity", "latitude", "longitude"},
    "incidents": {"incident_id", "location_id", "timestamp", "incident_type", "severity", "injuries", "reported_by", "resolved", "resolution_hours"},
    "complaints": {"complaint_id", "location_id", "created_at", "category", "priority_reported", "status", "days_to_resolution"},
}


@lru_cache(maxsize=1)
def load_data() -> dict[str, pd.DataFrame]:
    missing_files = [path.name for path in DATA_FILES.values() if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(f"Required dataset files are missing from {ROOT}: {', '.join(missing_files)}")
    frames = {name: pd.read_csv(path) for name, path in DATA_FILES.items()}
    for name, required in REQUIRED.items():
        missing = required - set(frames[name].columns)
        if missing:
            raise ValueError(f"{DATA_FILES[name].name} is missing required columns: {', '.join(sorted(missing))}")
    frames["incidents"]["timestamp"] = pd.to_datetime(frames["incidents"]["timestamp"], errors="coerce")
    frames["complaints"]["created_at"] = pd.to_datetime(frames["complaints"]["created_at"], errors="coerce")
    numeric_columns = {
        "locations": ["population_density_per_km2", "latitude", "longitude"],
        "incidents": ["injuries", "resolution_hours"],
        "complaints": ["days_to_resolution"],
    }
    for name, columns in numeric_columns.items():
        for column in columns:
            frames[name][column] = pd.to_numeric(frames[name][column], errors="coerce")
    for frame in frames.values():
        frame["location_id"] = frame["location_id"].astype(str)
    frames["incidents"]["resolved"] = frames["incidents"]["resolved"].astype(str).str.lower().isin({"1", "true", "yes"})
    for col in ("near_school", "near_hospital"):
        frames["locations"][col] = pd.to_numeric(frames["locations"][col], errors="coerce").fillna(0).astype(bool)
    known = set(frames["locations"]["location_id"])
    for name in ("incidents", "complaints"):
        unknown = sorted(set(frames[name]["location_id"]) - known)
        if unknown:
            raise ValueError(f"{name} contains {len(unknown)} location_id values absent from locations.csv")
    return frames


def data_quality() -> dict:
    frames = load_data()
    summaries = {}
    for name, frame in frames.items():
        identifier = next((column for column in frame.columns if column.endswith("_id") and column != "location_id"), None)
        date_column = "timestamp" if name == "incidents" else "created_at" if name == "complaints" else None
        invalid = {}
        if date_column:
            invalid[date_column] = int(frame[date_column].isna().sum())
        for column in frame.select_dtypes(include="number").columns:
            invalid[column] = int(frame[column].isna().sum())
        summaries[name] = {
            "records": len(frame),
            "columns": list(frame.columns),
            "missing": {column: int(value) for column, value in frame.isna().sum().items()},
            "duplicate_ids": int(frame[identifier].duplicated().sum()) if identifier else 0,
            "duplicate_location_ids": int(frame["location_id"].duplicated().sum()) if name == "locations" else None,
            "invalid_values": invalid,
            "locations_covered": int(frame["location_id"].nunique()),
        }
    return {
        "datasets": summaries,
        "join_key": "location_id",
        "location_count": len(frames["locations"]),
        "incident_join_coverage": round(frames["incidents"]["location_id"].nunique() / max(1, len(frames["locations"])), 4),
        "complaint_join_coverage": round(frames["complaints"]["location_id"].nunique() / max(1, len(frames["locations"])), 4),
        "incident_date_range": [str(frames["incidents"]["timestamp"].min().date()), str(frames["incidents"]["timestamp"].max().date())],
        "complaint_date_range": [str(frames["complaints"]["created_at"].min().date()), str(frames["complaints"]["created_at"].max().date())],
    }