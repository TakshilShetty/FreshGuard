"""SPORE//WATCH — UI theme constants and demo-mode truth table.

Kept separate from ``config.py`` so this stage never clobbers shared project
configuration owned by other modules.
"""
from __future__ import annotations

from pathlib import Path

# ---- palette (each colour has exactly one job) ------------------------------
CREAM, INK, IVORY = "#F5EFE2", "#17140D", "#FFFDF7"
TEAL, CORAL, MUSTARD, VIOLET, SKY = "#12B3A4", "#FF5B57", "#FFC531", "#6B5BE6", "#3AA0FF"
BG, PANEL, PANEL_2, TEXT, MUTED, BORDER = CREAM, IVORY, IVORY, INK, "#6F6A60", INK
LIME, CYAN, AMBER, CRIMSON = TEAL, SKY, MUSTARD, CORAL   # legacy tone names

# ---- identity ---------------------------------------------------------------
PRODUCT = "SporeWatch"
DIVISION = "BIOLOGICAL INTELLIGENCE DIVISION"
DESCRIPTOR = "BANANA FRESHNESS ANALYSIS SYSTEM"

# ---- demo-mode identifiers --------------------------------------------------
SPECIMEN_ID = "BANANA-001"
EXPERIMENT_ID = "EXP-DEMO-001"
MONITORING_STATUS = "DEMO ENVIRONMENT"
DATA_SOURCE = "SIMULATED"

# ---- demo dataset parameters (deterministic) --------------------------------
DEMO_SEED = 20240611
DEMO_ROWS = 24
DEMO_START = "2025-01-01 08:00:00"   # fixed anchor: synthetic, never "now"
DEMO_INTERVAL_MIN = 30

# ---- truthful initial system state (updated by later stages) ----------------
# tone: lime = active, cyan = info, amber = pending/caution, crimson = absent
SYSTEM_STATE: list[tuple[str, str, str]] = [
    ("HARDWARE", "NOT CONNECTED", "crimson"),
    ("DATA SOURCE", "SIMULATED", "cyan"),
    ("DATASET", "NO REAL DATASET", "amber"),
    ("ML MODEL", "NOT TRAINED", "crimson"),
    ("EXPERIMENTAL VALIDATION", "PENDING", "amber"),
]

MODEL_TRAINED = False   # flipped by a later stage when a validated model exists

CSS_PATH = Path(__file__).resolve().parent.parent / "assets" / "spore_watch.css"
