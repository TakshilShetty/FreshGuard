"""AI hook. predict(df) -> rich prediction dict or None.

Integrates with src.ai_engine to support:
- 3-class classification (FRESH/RIPENING/SPOILED)
- Shelf-life estimation in days/hours
- Risk scoring (0-100, LOW/MEDIUM/HIGH/CRITICAL)
- Trend analysis and early-warning alerts
- Hot-reloading of collaborator models from models/model.joblib
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.ai_engine import (
    MODELS_DIR,
    PRIMARY_MODEL_PATH,
    STANDARD_FEATURES,
    get_model_file,
    predict_dataframe,
)

MODEL_PATH = PRIMARY_MODEL_PATH
FEATURES = STANDARD_FEATURES


def predict(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Predict spoilage stage, confidence, risk, shelf-life for the latest row of df."""
    if df.empty:
        return None
    return predict_dataframe(df)
