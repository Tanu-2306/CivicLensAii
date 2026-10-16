from fastapi.testclient import TestClient

from backend.app.main import app


def test_random_forest_metrics_prediction_and_validation():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_available"] is True

        metrics_response = client.get("/model/metrics")
        assert metrics_response.status_code == 200
        metrics = metrics_response.json()
        assert metrics["model_name"] == "RandomForestRegressor"
        assert metrics["training_dataset_size"] == 7126
        assert metrics["held_out_test"]["random_forest"]["mae"] > 0
        assert metrics["cross_validation"]["method"].startswith("TimeSeriesSplit")
        assert metrics["feature_importance"]["by_source_feature"]["injuries"] >= 0
        assert metrics["target_distribution"]["count"] == metrics["held_out_test"]["median_baseline"]["count"] if "count" in metrics["held_out_test"]["median_baseline"] else metrics["target_distribution"]["count"] > 0
        assert metrics["prediction_distribution"]["unique_rounded_2dp"] > 1

        payload = {"location_id": "L0001", "incident_type": "Collision", "severity": "Critical", "reported_by": "Citizen", "reported_injuries": 2}
        prediction_response = client.post("/predict", json=payload)
        assert prediction_response.status_code == 200
        prediction = prediction_response.json()
        assert prediction["model"] == "RandomForestRegressor"
        assert prediction["unit"] == "hours"
        assert prediction["estimated_resolution_hours"] >= 0
        assert "not a guarantee" in prediction["disclaimer"]
        assert prediction["priority_recommendation"]["uses_predicted_duration"] is False

        out_of_support = client.post("/predict", json={**payload, "reported_injuries": 30})
        assert out_of_support.status_code == 200
        out_of_support_body = out_of_support.json()
        assert out_of_support_body["input_support_warnings"]
        assert out_of_support_body["priority_recommendation"]["level"] == "urgent_human_assessment"
        assert "resolution duration" in " ".join(out_of_support_body["limitations"])

        invalid_category = client.post("/predict", json={**payload, "severity": "Extreme"})
        assert invalid_category.status_code == 422
        unknown_location = client.post("/predict", json={**payload, "location_id": "UNKNOWN"})
        assert unknown_location.status_code == 422


def test_random_forest_cors_allows_configured_frontend_origin():
    with TestClient(app) as client:
        response = client.options("/predict", headers={"Origin": "http://127.0.0.1:5173", "Access-Control-Request-Method": "POST"})
        assert response.status_code == 200
        assert response.headers["access-control-allow-origin"] == "http://127.0.0.1:5173"