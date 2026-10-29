"""FreshGuard (SporeWatch) — Banana Spoilage Monitoring & AI System.

Public APIs and sub-packages.
"""
from __future__ import annotations

from src.data_manager import load_dataset, mark_as_synthetic, save_dataset
from src.data_validator import validate_sensor_data
from src.predictor import predict
from src.sensor_reader import build_dataframe, parse_line, read_samples

__all__ = [
    "load_dataset",
    "save_dataset",
    "mark_as_synthetic",
    "validate_sensor_data",
    "parse_line",
    "read_samples",
    "build_dataframe",
    "predict",
]
