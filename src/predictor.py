"""AI hook. predict(df) -> {"label": str, "confidence": float, ...} or None when no trained model exists.

Integrates with src.ai_engine to support collaborator-trained models in models/model.joblib,
as well as dynamic hot-reloading and fallback baseline heuristics.
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
    """Predict spoilage stage and confidence for the latest row of df."""
    if df.empty:
        return None
    return predict_dataframe(df)
