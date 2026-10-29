import pandas as pd
from src.data_manager import save_dataset
from src.data_validator import validate_sensor_data
from src.sensor_reader import build_dataframe, parse_line


def test_parse_good_line():
    assert parse_line("1850,26.4,61.2\r\n") == {"gas_raw": 1850, "temperature_c": 26.4, "humidity_percent": 61.2}


def test_parse_rejects_garbage():
    for bad in ["", "gas_raw,temperature_c,humidity_percent", "1,2", "a,b,c", "9999,25,50", "1800,25,150", "1800,nan,50"]:
        assert parse_line(bad) is None


def test_real_rows_validate_and_save(tmp_path):
    rows = [{"timestamp": f"2026-10-29 10:00:{i:02d}", "gas_raw": 1800 + i, "temperature_c": 26.0, "humidity_percent": 60.0} for i in range(5)]
    df = build_dataframe(rows, "BANANA-001")
    assert "spoilage_label" not in df.columns
    assert validate_sensor_data(df, expected_origin="real").is_valid
    assert save_dataset(df, tmp_path / "data" / "raw" / "t.csv", origin="real").saved
