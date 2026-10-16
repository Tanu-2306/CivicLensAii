from fastapi.testclient import TestClient

from backend.app.data import data_quality, load_data
from backend.app.main import _priority_rows, app


def test_supplied_data_join_coverage_and_quality():
    data = load_data()
    report = data_quality()
    assert len(data["locations"]) == 1000
    assert len(data["incidents"]) == 10000
    assert len(data["complaints"]) == 8000
    assert report["incident_join_coverage"] == 1
    assert report["complaint_join_coverage"] == 1
    assert report["datasets"]["incidents"]["duplicate_ids"] == 0
    assert report["datasets"]["complaints"]["duplicate_ids"] == 0


def test_priority_aggregation_preserves_location_count_and_record_totals():
    rows = _priority_rows()
    assert len(rows) == 1000
    assert sum(row["incident_count"] for row in rows) == 10000
    assert sum(row["complaint_count"] for row in rows) == 8000
    assert all(0 <= row["priority_score"] <= 100 for row in rows)
    assert all(set(row["components"]) == {"incident_volume", "severity", "injury_burden", "unresolved_incidents", "open_complaints", "road_condition", "resolution_delay", "context"} for row in rows)


def test_api_reads_real_records_and_serves_scratch_model():
    with TestClient(app) as client:
        assert client.get("/health").json()["synthetic"] is True
        dashboard = client.get("/api/dashboard").json()
        assert dashboard["locations"] == 1000
        assert dashboard["incidents"] == 10000
        assert client.get("/api/locations/L0001").status_code == 200
        detail = client.get("/api/locations/L0001").json()
        assert detail["prediction"]["incident_resolution_hours"] is not None
        status = client.get("/api/models/status").json()
        assert status["artifact_available"] is True
        assert status["model_type"] == "NumPy gradient-descent linear regression"
        assert status["test_records"] == 1426
        selected = status["selected_predictor"]
        assert selected["mae_by_candidate"][selected["name"]] == min(selected["mae_by_candidate"].values())
        metrics = client.get("/api/models/metrics").json()
        assert metrics["manual_linear_regression"]["manual_linear_regression"]["metrics"]["mse"] > 0
        history = client.get("/api/models/training-history").json()
        assert history["count"] == status["training"]["epochs_completed"]
        prediction = client.post("/api/models/predict", json={"location_id": "L0001"})
        assert prediction.status_code == 200
        assert prediction.json()["unit"] == "hours"
        assert prediction.json()["estimate"] >= 0
        assert prediction.json()["scratch_linear_regression_estimate"] >= 0
        assert client.get("/api/incidents/trends?period=month").status_code == 200
        assert client.get("/api/locations/UNKNOWN").status_code == 404