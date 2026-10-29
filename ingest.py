"""ESP32 serial -> MongoDB freshguard.readings (one doc per valid line).

  python ingest.py [--port COM5] [--specimen BANANA-LIVE]     Ctrl+C stops.
"""
import argparse
import os
from datetime import datetime

import serial
from pymongo import MongoClient

from src import sensor_reader as sr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port"); ap.add_argument("--specimen", default="BANANA-LIVE"); ap.add_argument("--baud", type=int, default=115200)
    a = ap.parse_args()
    if not a.port:
        ports = [p.split()[0] for p in sr.list_ports()]
        if len(ports) != 1:
            raise SystemExit(f"Pass --port; found: {ports or 'none (check USB cable/driver)'}")
        a.port = ports[0]
    col = MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017")).freshguard.readings
    print(f"Ingesting {a.port} -> freshguard.readings as {a.specimen}. Ctrl+C to stop.")
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
