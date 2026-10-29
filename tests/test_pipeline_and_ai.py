import pandas as pd
import pytest

from src import ai_engine
from src.pipeline import LivePipeline


def test_ai_engine_baseline_heuristics():
    fresh = ai_engine.compute_baseline_prediction(900, 25.0, 50.0)
    assert fresh["label"] == "fresh"
    assert fresh["risk_score"] < 30.0

    ripe = ai_engine.compute_baseline_prediction(1800, 26.0, 55.0)
    assert ripe["label"] == "ripe"
    assert 30.0 <= ripe["risk_score"] <= 75.0

    spoiled = ai_engine.compute_baseline_prediction(2600, 28.0, 65.0)
    assert spoiled["label"] == "spoiled"
    assert spoiled["risk_score"] > 70.0


def test_ai_engine_predict_dataframe():
    df = pd.DataFrame([
        {"gas_raw": 1000, "temperature_c": 25.0, "humidity_percent": 50.0},
        {"gas_raw": 2800, "temperature_c": 30.0, "humidity_percent": 70.0},
    ])
    res = ai_engine.predict_dataframe(df)
    assert res is not None
    assert "label" in res
    assert "confidence" in res
    assert "risk_score" in res


def test_pipeline_process_frame_in_memory():
    pipeline = LivePipeline(specimen_id="TEST-SPECIMEN-01", mongo_uri="mongodb://invalid_uri_to_test_fallback:9999/")
    sample = {"gas_raw": 1150, "temperature_c": 27.5, "humidity_percent": 52.0}

    doc = pipeline.process_frame(sample)
    assert doc["specimen_id"] == "TEST-SPECIMEN-01"
    assert doc["gas_raw"] == 1150
    assert doc["temperature_c"] == 27.5
    assert doc["humidity_percent"] == 52.0
    assert "spoilage_label" in doc
    assert "confidence" in doc
    assert "processed" in doc
    assert len(pipeline.recent_readings) == 1


def test_ai_engine_trained_model_inference():
    """Verify inference with the trained model bundle."""
    # Fresh banana input
    fresh_res = ai_engine.predict_sample({"gas_raw": 350, "temperature_c": 26.0, "humidity_percent": 55.0})
    assert fresh_res["label"] == "FRESH"
    assert fresh_res["confidence"] > 0.60
    assert "shelf_life_days" in fresh_res
    assert fresh_res["shelf_life_days"] > 2.0
    assert fresh_res["risk_level"] in ("LOW", "MEDIUM")

    # Spoiled banana input
    spoil_res = ai_engine.predict_sample({"gas_raw": 2200, "temperature_c": 29.0, "humidity_percent": 68.0})
    assert spoil_res["label"] == "SPOILED"
    assert spoil_res["confidence"] > 0.80
    assert spoil_res["shelf_life_days"] < 1.5
    assert spoil_res["risk_level"] in ("HIGH", "CRITICAL")
    assert len(spoil_res["alerts"]) > 0


def test_trend_tracker_acceleration():
    """Verify sliding-window trend tracker calculates rate and deterioration."""
    tracker = ai_engine.TrendTracker(window_size=10)
    for g in [500, 520, 550, 600, 700, 850, 1050, 1300]:
        tracker.add({"gas_raw": g, "temperature_c": 26.0, "humidity_percent": 55.0})

    trend = tracker.get_trend()
    assert trend["samples"] == 8
    assert trend["gas_trend"] in ("RISING", "RISING_FAST")
    assert trend["gas_rate"] > 0
    assert trend["deterioration_speed"] in ("ACCELERATING", "MODERATE")


def test_server_api_endpoints():
    """Test FastAPI REST endpoints."""
    from fastapi.testclient import TestClient
    from src.server import app

    client = TestClient(app)

    # 1. Root
    r_root = client.get("/")
    assert r_root.status_code == 200
    assert "capabilities" in r_root.json()

    # 2. Status
    r_status = client.get("/api/status")
    assert r_status.status_code == 200
    assert r_status.json()["ai_engine"]["available"] is True

    # 3. Model info
    r_info = client.get("/api/model/info")
    assert r_info.status_code == 200
    assert r_info.json()["loaded"] is True
    assert "classifier" in r_info.json()
    assert "regressor" in r_info.json()

    # 4. Predict endpoint
    r_pred = client.post(
        "/api/predict",
        json={"gas_raw": 1950.0, "temperature_c": 28.5, "humidity_percent": 65.0},
    )
    assert r_pred.status_code == 200
    p = r_pred.json()["prediction"]
    assert p["label"] == "SPOILED"
    assert p["risk_score"] > 60.0
