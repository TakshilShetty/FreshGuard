"""Collect REAL readings from the ESP32 and save them to data/raw/.

Usage:
  python scripts/collect_data.py --list-ports
  python scripts/collect_data.py --port COM5 --specimen BANANA-001 --minutes 30
  python scripts/collect_data.py --port COM5 --specimen BANANA-001 --minutes 30 --label ripe

Ctrl+C stops early and still saves what was read.
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

from src import sensor_reader as sr
from src.config import DATA_DIR, DEFAULT_BAUD, DEFAULT_SPECIMEN
from src.data_manager import save_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect real sensor readings from ESP32 into data/raw/")
    ap.add_argument("--list-ports", action="store_true", help="List available serial COM ports")
    ap.add_argument("--port", help="Serial port of ESP32 (e.g. COM3, COM5, /dev/ttyUSB0)")
    ap.add_argument("--specimen", default=DEFAULT_SPECIMEN, help=f"Specimen identifier (default: {DEFAULT_SPECIMEN})")
    ap.add_argument("--minutes", type=float, default=10, help="Duration to record in minutes (default: 10)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Serial baud rate (default: {DEFAULT_BAUD})")
    ap.add_argument("--label", default=None, help="Human observation label (e.g. fresh / ripe / spoiled)")
    a = ap.parse_args()

    if a.list_ports:
        print("\n".join(sr.list_ports()) or "No serial ports found.")
        return
    if not a.port:
        ap.error("--port is required (use --list-ports to find it)")

    print(f"Reading {a.port} for {a.minutes} min... Ctrl+C to stop early.")
    rows = sr.read_samples(
        a.port,
        a.baud,
        a.minutes * 60,
        lambda r, n: print(f"{n:4d}  gas={r['gas_raw']}  {r['temperature_c']}C  {r['humidity_percent']}%"),
    )
    df = sr.build_dataframe(rows, a.specimen, a.label)
    if df.empty:
        print("No valid readings received. Check port, baud rate and wiring.")
        return

    dest = DATA_DIR / "raw" / f"{a.specimen}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    res = save_dataset(df, dest, origin="real")
    print("Saved" if res.saved else "NOT saved", dest, "|", len(df), "rows")
    for msg in list(getattr(res, "errors", [])) + list(getattr(res, "warnings", [])):
        print(" -", msg)


if __name__ == "__main__":
    main()
