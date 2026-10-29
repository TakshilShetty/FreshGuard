"""FreshGuard global configuration and environment defaults."""
from __future__ import annotations

import os
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# MongoDB defaults
MONGO_URI = os.environ.get("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.environ.get("MONGO_DB", "freshguard")
READINGS_COLLECTION = "readings"

# Web dashboard & network defaults
PORT = int(os.environ.get("PORT", "5000"))
MDNS_NAME = "freshguard"

# Hardware / serial defaults
DEFAULT_BAUD = int(os.environ.get("SERIAL_BAUD", "115200"))
DEFAULT_SPECIMEN = "BANANA-LIVE"
