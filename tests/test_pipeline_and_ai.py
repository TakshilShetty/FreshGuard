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
