"""Load a CSV into MongoDB. Usage: python scripts/load_to_mongo.py [csv_path]"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from pymongo import MongoClient

from src.config import DB_NAME, MONGO_URI, READINGS_COLLECTION


def main() -> None:
    ap = argparse.ArgumentParser(description="Load a sensor dataset CSV into MongoDB collection")
    ap.add_argument("csv_path", nargs="?", default="data/demo/demo_5day_3bananas.csv", help="Path to CSV dataset")
    ap.add_argument("--append", action="store_true", help="Append records instead of replacing collection")
    a = ap.parse_args()

    csv_file = Path(a.csv_path)
    if not csv_file.is_file():
        csv_file = ROOT / a.csv_path
        if not csv_file.is_file():
            sys.exit(f"Error: File not found: {a.csv_path}")

    df = pd.read_csv(csv_file, parse_dates=["timestamp"])
    col = MongoClient(MONGO_URI)[DB_NAME][READINGS_COLLECTION]
    if not a.append:
        col.drop()
    col.insert_many(df.to_dict("records"))
    print(f"Loaded {len(df)} rows from {csv_file.name}. Total in {DB_NAME}.{READINGS_COLLECTION}: {col.count_documents({})}")


if __name__ == "__main__":
    main()
