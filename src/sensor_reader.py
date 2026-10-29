"""Read ESP32 serial lines and turn them into a validator-ready DataFrame.

Line format sent by firmware/esp32_sensor: ``gas_raw,temperature_c,humidity_percent``
or formatted multi-line test stream (MQ-3 Sensor Value: ..., Temperature: ..., Humidity: ...).
Nothing here invents data: unreadable lines are skipped, never guessed.
"""
from __future__ import annotations

import math
import re
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


class StreamParser:
    """Stateful stream parser that processes both CSV and multi-line formatted serial streams."""

    def __init__(self) -> None:
        self._buffer: dict[str, float | int] = {}

    def feed(self, line: str) -> Optional[dict]:
        # 1. Try single-line CSV format first
        direct = parse_line(line)
        if direct is not None:
            return direct

        # 2. Key-value multi-line stream pattern matching
        mq_m = re.search(r"MQ[-_ ]?3(?:.*?Value)?\s*[:=]\s*(\d+)", line, re.IGNORECASE)
        if mq_m:
            self._buffer["gas_raw"] = int(mq_m.group(1))

        t_m = re.search(r"Temp(?:erature)?\s*[:=]\s*([0-9.]+)", line, re.IGNORECASE)
        if t_m:
            self._buffer["temperature_c"] = float(t_m.group(1))

        h_m = re.search(r"Hum(?:idity)?\s*[:=]\s*([0-9.]+)", line, re.IGNORECASE)
        if h_m:
            self._buffer["humidity_percent"] = float(h_m.group(1))

        if "gas_raw" in self._buffer and "temperature_c" in self._buffer and "humidity_percent" in self._buffer:
            g = self._buffer["gas_raw"]
            t = self._buffer["temperature_c"]
            h = self._buffer["humidity_percent"]
            self._buffer.clear()
            if 0 <= g <= ADC_MAX and 0 <= h <= 100 and math.isfinite(t):
                return {"gas_raw": int(g), "temperature_c": t, "humidity_percent": h}
        return None


def list_ports() -> list[str]:
    from serial.tools import list_ports as lp
    return [f"{p.device}  ({p.description})" for p in lp.comports()]


def read_samples(port: str, baud: int = 115200, seconds: float = 600, on_sample: Optional[Callable] = None) -> list[dict]:
    """Collect readings for ``seconds`` (Ctrl+C stops early and keeps what was read)."""
    import serial  # pyserial, imported lazily so tests need no hardware

    parser = StreamParser()
    samples: list[dict] = []
    end = datetime.now().timestamp() + seconds
    try:
        with serial.Serial(port, baud, timeout=3) as ser:
            ser.reset_input_buffer()
            while datetime.now().timestamp() < end:
                row = parser.feed(ser.readline().decode("utf-8", errors="ignore"))
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
