"""Collect REAL readings from the ESP32 and save them to data/raw/.

  python collect_data.py --list-ports
  python collect_data.py --port COM5 --specimen BANANA-001 --minutes 30
  python collect_data.py --port COM5 --specimen BANANA-001 --minutes 30 --label ripe   (label only if YOU judged it)
Ctrl+C stops early and still saves what was read.
"""
import argparse
from datetime import datetime
from pathlib import Path

from src import sensor_reader as sr
from src.data_manager import save_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-ports", action="store_true")
    ap.add_argument("--port"); ap.add_argument("--specimen", default="BANANA-001")
    ap.add_argument("--minutes", type=float, default=10); ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--label", default=None, help="your own human label, e.g. fresh / ripe / spoiled")
    a = ap.parse_args()
    if a.list_ports:
        print("\n".join(sr.list_ports()) or "No serial ports found."); return
    if not a.port:
        ap.error("--port is required (try --list-ports)")
    print(f"Reading {a.port} for {a.minutes} min... Ctrl+C to stop early.")
    rows = sr.read_samples(a.port, a.baud, a.minutes * 60, lambda r, n: print(f"{n:4d}  gas={r['gas_raw']}  {r['temperature_c']}C  {r['humidity_percent']}%"))
    df = sr.build_dataframe(rows, a.specimen, a.label)
    if df.empty:
        print("No valid readings received. Check port, baud rate and wiring."); return
    dest = Path("data/raw") / f"{a.specimen}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    res = save_dataset(df, dest, origin="real")
    print("Saved" if res.saved else "NOT saved", dest, "|", len(df), "rows")
    for msg in list(getattr(res, "errors", [])) + list(getattr(res, "warnings", [])):
        print(" -", msg)


if __name__ == "__main__":
    main()
