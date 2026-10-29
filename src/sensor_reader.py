"""Read ESP32 serial lines and turn them into a validator-ready DataFrame.

Line format sent by firmware/esp32_sensor: ``gas_raw,temperature_c,humidity_percent``.
Nothing here invents data: unreadable lines are skipped, never guessed.
"""
from __future__ import annotations

import math
from datetime import datetime
from typing import Callable, Optional

import pandas as pd

SOURCE_LABEL = "esp32_mq3"
ADC_MAX = 4095


def parse_line(line: str) -> Optional[dict]:
    """Return {'gas_raw','temperature_c','humidity_percent'} or None if the line is not a reading."""
    parts = line.strip().split(",")
    if len(parts) != 3:
        return None
    try:
        gas, temp, hum = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (gas, temp, hum)) or not 0 <= gas <= ADC_MAX or not 0 <= hum <= 100:
        return None
    return {"gas_raw": int(round(gas)), "temperature_c": temp, "humidity_percent": hum}


def list_ports() -> list[str]:
    from serial.tools import list_ports as lp
    return [f"{p.device}  ({p.description})" for p in lp.comports()]


def read_samples(port: str, baud: int = 115200, seconds: float = 600, on_sample: Optional[Callable] = None) -> list[dict]:
    """Collect readings for ``seconds`` (Ctrl+C stops early and keeps what was read)."""
    import serial  # pyserial, imported lazily so tests need no hardware

    samples: list[dict] = []
    end = datetime.now().timestamp() + seconds
    try:
        with serial.Serial(port, baud, timeout=3) as ser:
            ser.reset_input_buffer()
            while datetime.now().timestamp() < end:
                row = parse_line(ser.readline().decode("utf-8", errors="ignore"))
                if row:
                    row["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    samples.append(row)
                    if on_sample:
                        on_sample(row, len(samples))
    except KeyboardInterrupt:
        pass
    return samples


def build_dataframe(samples: list[dict], specimen_id: str, label: Optional[str] = None) -> pd.DataFrame:
    """Samples -> DataFrame using the validator's column names. Label is only added if YOU give one."""
    df = pd.DataFrame(samples)
    if df.empty:
        return df
    stamp = pd.to_datetime(df["timestamp"]).dt.strftime("%Y%m%d%H%M%S")
    df["sample_id"] = [f"{specimen_id}-{s}-{i:04d}" for i, s in enumerate(stamp)]
    df["specimen_id"] = specimen_id
    df["source"] = SOURCE_LABEL
    cols = ["timestamp", "sample_id", "specimen_id", "gas_raw", "temperature_c", "humidity_percent", "source"]
    if label:
        df["spoilage_label"] = label
        cols.append("spoilage_label")
    return df[cols]
