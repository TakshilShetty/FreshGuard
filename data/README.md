# FreshGuard Data Tiering & Provenance

This directory implements a strict three-tier architecture enforced by [`src/data_manager.py`](../src/data_manager.py) and [`src/data_validator.py`](../src/data_validator.py).

```text
data/
├── demo/          # Synthetic simulation data for demonstrations and offline tests
├── raw/           # Unmodified time-series recordings straight from the ESP32
└── processed/     # Cleaned, standardized, feature-engineered datasets for ML training
```

## Data Tiers

### 1. `data/demo/`
- **Purpose**: Fully synthetic deterministic datasets used when hardware/MongoDB is unavailable.
- **Rules**: Every record must carry a `source` column marked as `synthetic` (enforced by `mark_as_synthetic()`).
- **Included files**:
  - `demo_5day_3bananas.csv`: 5-day continuous decay trajectory across 3 bananas and an empty control chamber.

### 2. `data/raw/`
- **Purpose**: Authentic sensor readings collected from the hardware.
- **Rules**:
  - Generated using `python scripts/collect_data.py`.
  - Stored with timestamped filenames: `<specimen_id>_YYYYMMDD_HHMMSS.csv`.
  - **Strict provenance**: Synthetic data is strictly forbidden from being written to `data/raw/` by default.

### 3. `data/processed/`
- **Purpose**: Curated training and validation sets for model fitting.
- **Rules**: Verified by `validate_sensor_data()`, missing values handled, and scaled/normalized for machine learning pipelines.
