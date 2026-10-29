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
    """Return dict with sensor readings or None if the line is not a reading.
    
    Supports:
    - 3-field standard: gas_raw,temperature_c,humidity_percent
    - 7-field IMU: gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g
    - 9-field IMU with tilt: gas_raw,temperature_c,humidity_percent,accel_x,accel_y,accel_z,shock_g,pitch_deg,roll_deg
    """
    parts = line.strip().split(",")
    if len(parts) not in (3, 7, 9):
        return None
    try:
        gas, temp, hum = float(parts[0]), float(parts[1]), float(parts[2])
    except ValueError:
        return None
    if not all(math.isfinite(v) for v in (gas, temp, hum)) or not 0 <= gas <= ADC_MAX or not 0 <= hum <= 100:
        return None
    
    data: dict[str, Any] = {"gas_raw": int(round(gas)), "temperature_c": temp, "humidity_percent": hum}
    if len(parts) >= 7:
        try:
            ax, ay, az, shock = float(parts[3]), float(parts[4]), float(parts[5]), float(parts[6])
            if all(math.isfinite(v) for v in (ax, ay, az, shock)):
                data.update({"accel_x": ax, "accel_y": ay, "accel_z": az, "shock_g": shock})
        except ValueError:
            pass
    if len(parts) == 9:
        try:
            pitch, roll = float(parts[7]), float(parts[8])
            if all(math.isfinite(v) for v in (pitch, roll)):
                data.update({"pitch_deg": pitch, "roll_deg": roll})
        except ValueError:
            pass
    return data


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

        shock_m = re.search(r"(?:Shock|Impact|Accel(?:eration)?)\s*[:=]\s*([0-9.]+)", line, re.IGNORECASE)
        if shock_m:
            self._buffer["shock_g"] = float(shock_m.group(1))

        if "gas_raw" in self._buffer and "temperature_c" in self._buffer and "humidity_percent" in self._buffer:
            g = self._buffer["gas_raw"]
            t = self._buffer["temperature_c"]
            h = self._buffer["humidity_percent"]
            shock = self._buffer.get("shock_g")
            self._buffer.clear()
            if 0 <= g <= ADC_MAX and 0 <= h <= 100 and math.isfinite(t):
                row = {"gas_raw": int(g), "temperature_c": t, "humidity_percent": h}
                if shock is not None and math.isfinite(shock):
                    row["shock_g"] = float(shock)
                return row
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
    for extra in ["accel_x", "accel_y", "accel_z", "shock_g", "pitch_deg", "roll_deg"]:
        if extra in df.columns:
            cols.append(extra)
    if label and "spoilage_label" in df.columns:
        cols.append("spoilage_label")
    return df[cols]
