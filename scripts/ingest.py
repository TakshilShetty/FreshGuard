"""ESP32 serial -> MongoDB freshguard.readings (one doc per valid line).

Usage:
  python scripts/ingest.py [--port COM5] [--specimen BANANA-LIVE] [--baud 115200]
Ctrl+C stops.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import serial
from pymongo import MongoClient

from src import sensor_reader as sr
from src.config import DB_NAME, DEFAULT_BAUD, DEFAULT_SPECIMEN, MONGO_URI, READINGS_COLLECTION


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream ESP32 serial sensor readings into MongoDB")
    ap.add_argument("--port", help="Serial port of ESP32 (auto-detected if only one is connected)")
    ap.add_argument("--specimen", default=DEFAULT_SPECIMEN, help=f"Specimen label (default: {DEFAULT_SPECIMEN})")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Serial baud rate (default: {DEFAULT_BAUD})")
    a = ap.parse_args()

    if not a.port:
        ports = [p.split()[0] for p in sr.list_ports()]
        if len(ports) != 1:
            raise SystemExit(f"Pass --port; found: {ports or 'none (check USB cable/driver)'}")
        a.port = ports[0]

    col = MongoClient(MONGO_URI)[DB_NAME][READINGS_COLLECTION]
    print(f"Ingesting {a.port} -> {DB_NAME}.{READINGS_COLLECTION} as {a.specimen}. Ctrl+C to stop.")
    n = 0
    try:
        with serial.Serial(a.port, a.baud, timeout=3) as ser:
            ser.reset_input_buffer()
            while True:
                row = sr.parse_line(ser.readline().decode("utf-8", errors="ignore"))
                if row:
                    col.insert_one({"timestamp": datetime.now(), "specimen_id": a.specimen, **row, "source": sr.SOURCE_LABEL})
                    n += 1
                    print(f"{n:5d}  gas={row['gas_raw']}  {row['temperature_c']}C  {row['humidity_percent']}%")
    except KeyboardInterrupt:
        print(f"\nStopped. {n} readings stored.")


if __name__ == "__main__":
    main()
