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


def test_transit_tracker_shocks_and_tilt():
    """Verify TransitTracker computes peak shock, drops, tilt violations, and bruise penalties."""
    tracker = ai_engine.TransitTracker(window_size=10)

    # Normal resting readings
    for _ in range(5):
        tracker.add({"accel_x": 0.01, "accel_y": 0.02, "accel_z": 0.99, "shock_g": 1.0})

    state = tracker.get_transit_state()
    assert state["live_shock_g"] == 1.0
    assert state["tilt_status"] == "LEVEL & UPRIGHT"
    assert state["severe_drops"] == 0
    assert state["handling_integrity_score"] == 100.0
    assert state["bruise_penalty_percent"] == 0.0

    # Severe drop event (3.2g)
    tracker.add({"accel_x": 0.5, "accel_y": 0.2, "accel_z": 3.1, "shock_g": 3.2, "pitch_deg": 10.0, "roll_deg": 5.0})
    state_after_drop = tracker.get_transit_state()
    assert state_after_drop["severe_drops"] == 1
    assert state_after_drop["peak_shock_g"] == 3.2
    assert state_after_drop["handling_integrity_score"] < 100.0
    assert state_after_drop["bruise_penalty_percent"] > 0.0
    assert len(state_after_drop["incident_log"]) > 0
    assert state_after_drop["incident_log"][0]["type"] == "SEVERE_DROP"


def test_mpu6050_sensor_reader_parsing():
    """Verify sensor_reader parses 3-field, 7-field, and 9-field CSV lines."""
    from src.sensor_reader import parse_line

    # Standard 3-field
    r3 = parse_line("1200,25.5,55.0\n")
    assert r3 == {"gas_raw": 1200, "temperature_c": 25.5, "humidity_percent": 55.0}

    # 7-field with IMU
    r7 = parse_line("1200,25.5,55.0,0.02,-0.01,0.98,1.05\n")
    assert r7["gas_raw"] == 1200
    assert r7["shock_g"] == 1.05
    assert r7["accel_z"] == 0.98

    # 9-field with Pitch and Roll
    r9 = parse_line("1200,25.5,55.0,0.05,-0.02,0.99,1.02,2.8,-1.2\n")
    assert r9["pitch_deg"] == 2.8
    assert r9["roll_deg"] == -1.2


def test_dual_health_scores_and_bruise_adjusted_shelf_life():
    """Verify dual health score calculation and bruise shelf-life penalty."""
    sample = {
        "gas_raw": 400,
        "temperature_c": 24.0,
        "humidity_percent": 50.0,
        "accel_x": 0.0,
        "accel_y": 0.0,
        "accel_z": 3.5,
        "shock_g": 3.5,
    }
    res = ai_engine.predict_sample(sample)
    assert "bio_freshness_score" in res
    assert "handling_integrity_score" in res
    assert "composite_transit_score" in res
    assert "transit" in res
    assert res["transit"]["peak_shock_g"] >= 3.5
    # Shelf life adjusted due to 3.5g drop
    assert res["shelf_life_days"] <= res["shelf_life_days_baseline"]
    # Check that transit impact alert was generated
    impact_alerts = [a for a in res["alerts"] if "IMPACT" in a["message"] or "drop" in a["message"].lower()]
    assert len(impact_alerts) > 0
