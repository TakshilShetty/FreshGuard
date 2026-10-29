# FreshGuard CLI Scripts & Utilities

This folder contains command-line automation tools and data ingestion scripts for the FreshGuard system.

| Script | Purpose | Example Command |
| :--- | :--- | :--- |
| [`collect_data.py`](collect_data.py) | Streams live data from ESP32 to timestamped CSV under `data/raw/` | `python scripts/collect_data.py --port COM5 --specimen BANANA-001` |
| [`ingest.py`](ingest.py) | Continuous serial-to-MongoDB pipeline | `python scripts/ingest.py --port COM5 --specimen BANANA-LIVE` |
| [`load_to_mongo.py`](load_to_mongo.py) | Imports historical or demo CSV dataset into MongoDB | `python scripts/load_to_mongo.py data/demo/demo_5day_3bananas.csv` |
| [`make_demo_data.py`](make_demo_data.py) | Generates synthetic 5-day decay simulation dataset | `python scripts/make_demo_data.py` |

All scripts are configured to automatically resolve the project root and load environment defaults from [`src/config.py`](../src/config.py).
