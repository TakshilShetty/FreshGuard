"""Stream ESP32 serial sensor readings -> AI Engine -> MongoDB freshguard.readings.

Usage:
    python scripts/ingest.py [--port COM5] [--specimen BANANA-LIVE] [--baud 115200]
Ctrl+C stops.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import DEFAULT_BAUD, DEFAULT_SPECIMEN
from src.pipeline import LivePipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream ESP32 serial sensor readings through AI Engine into MongoDB")
    ap.add_argument("--port", help="Serial port of ESP32 (auto-detected if omitted)")
    ap.add_argument("--specimen", default=DEFAULT_SPECIMEN, help=f"Specimen label (default: {DEFAULT_SPECIMEN})")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Serial baud rate (default: {DEFAULT_BAUD})")
    a = ap.parse_args()

    pipeline = LivePipeline(port=a.port, baud=a.baud, specimen_id=a.specimen)
    pipeline.run()


if __name__ == "__main__":
    main()
