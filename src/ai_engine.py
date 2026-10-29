"""AI Engine for FreshGuard / SporeWatch banana freshness classification.

Handles model loading, feature engineering, and inference for both
collaborator-trained models (scikit-learn / joblib / pickle) and baseline heuristics.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PRIMARY_MODEL_PATH = MODELS_DIR / "model.joblib"
SECONDARY_MODEL_PATH = MODELS_DIR / "model.pkl"
STANDARD_FEATURES = ["gas_raw", "temperature_c", "humidity_percent"]

# In-memory cached model and its last modification timestamp
_CACHED_MODEL: Any = None
_CACHED_MTIME: float = 0.0
_CACHED_PATH: Optional[Path] = None


def get_model_file() -> Optional[Path]:
    """Return the active model path if one exists, prioritizing model.joblib then model.pkl."""
    if PRIMARY_MODEL_PATH.is_file():
        return PRIMARY_MODEL_PATH
    if SECONDARY_MODEL_PATH.is_file():
        return SECONDARY_MODEL_PATH
    return None


def load_collaborator_model() -> Tuple[Optional[Any], Optional[Path]]:
    """Hot-reload collaborator model if modified or newly created."""
    global _CACHED_MODEL, _CACHED_MTIME, _CACHED_PATH

    model_file = get_model_file()
    if model_file is None:
        _CACHED_MODEL = None
        _CACHED_MTIME = 0.0
        _CACHED_PATH = None
        return None, None

    try:
        mtime = model_file.stat().st_mtime
        if _CACHED_MODEL is None or _CACHED_PATH != model_file or mtime > _CACHED_MTIME:
            import joblib

            _CACHED_MODEL = joblib.load(model_file)
            _CACHED_MTIME = mtime
            _CACHED_PATH = model_file
            logger.info("Loaded collaborator model from %s", model_file.name)
    except Exception as exc:
        logger.warning("Failed to load collaborator model from %s: %s", model_file, exc)
        return None, None

    return _CACHED_MODEL, _CACHED_PATH


def extract_features(df_or_sample: pd.DataFrame | Dict[str, Any]) -> pd.DataFrame:
    """Ensure data has the required feature columns."""
    if isinstance(df_or_sample, dict):
        df = pd.DataFrame([df_or_sample])
    else:
        df = df_or_sample.copy()

    for col in STANDARD_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Missing required feature column: {col}")

    return df[STANDARD_FEATURES].astype(float)


def compute_baseline_prediction(gas_raw: float, temp_c: float, humidity: float) -> Dict[str, Any]:
    """Domain-heuristic baseline for banana spoilage based on gas accumulation.

    Used when collaborator model is being developed or not yet placed in models/.
    - Fresh bananas emit minimal volatile organic compounds (VOCs).
    - Ripening bananas produce increasing VOCs (ethylene and esters).
    - Overripe/spoiled bananas emit high gas counts (alcohols, acetic acid).
    """
    # Baseline MQ-3 response calibration thresholds
    if gas_raw < 1500:
        label = "fresh"
        base_conf = 0.90 - (gas_raw / 1500.0) * 0.15
        risk_score = (gas_raw / 1500.0) * 30.0
    elif gas_raw < 2300:
        label = "ripe"
        progress = (gas_raw - 1500.0) / (2300.0 - 1500.0)
        base_conf = 0.82 + progress * 0.10
        risk_score = 30.0 + progress * 40.0
    else:
        label = "spoiled"
        overflow = min(1.0, (gas_raw - 2300.0) / 1000.0)
        base_conf = 0.90 + overflow * 0.08
        risk_score = min(100.0, 70.0 + overflow * 30.0)

    # Temperature & humidity acceleration factors
    temp_penalty = max(0.0, (temp_c - 25.0) * 0.8)
    humidity_penalty = max(0.0, (humidity - 70.0) * 0.3)
    adjusted_risk = min(100.0, max(0.0, risk_score + temp_penalty + humidity_penalty))

    return {
        "label": label,
        "confidence": float(np.clip(base_conf, 0.50, 0.99)),
        "risk_score": round(float(adjusted_risk), 1),
        "model_source": "baseline_heuristic",
        "collaborator_status": "Awaiting collaborator model (drop model.joblib in models/)",
    }


def predict_dataframe(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Predict spoilage state from the latest row of a DataFrame."""
    if df.empty:
        return None

    model, path = load_collaborator_model()
    latest_row = df.iloc[-1]
    gas = float(latest_row["gas_raw"])
    temp = float(latest_row["temperature_c"])
    hum = float(latest_row["humidity_percent"])

    # If collaborator model exists, use it
    if model is not None:
        try:
            x = extract_features(df).tail(1)
            pred_label = str(model.predict(x)[0])
            conf: Optional[float] = None
            probabilities: Dict[str, float] = {}

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x)[0]
                classes = getattr(model, "classes_", [str(i) for i in range(len(probs))])
                probabilities = {str(c): round(float(p), 4) for c, p in zip(classes, probs)}
                conf = float(probs.max())

            # Estimate risk score from prediction
            risk_map = {"fresh": 15.0, "ripe": 50.0, "spoiled": 92.0, "no_banana": 0.0}
            risk_score = risk_map.get(pred_label, 50.0)
            if probabilities.get("spoiled"):
                risk_score = probabilities["spoiled"] * 100.0

            return {
                "label": pred_label,
                "confidence": conf,
                "risk_score": round(risk_score, 1),
                "model_source": "collaborator_model",
                "model_file": path.name if path else "model.joblib",
                "probabilities": probabilities,
                "features": {"gas_raw": gas, "temperature_c": temp, "humidity_percent": hum},
            }
        except Exception as exc:
            logger.warning("Collaborator model inference failed: %s; falling back to heuristic", exc)

    # Fallback to domain baseline heuristic
    res = compute_baseline_prediction(gas, temp, hum)
    res["features"] = {"gas_raw": gas, "temperature_c": temp, "humidity_percent": hum}
    return res


def predict_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Predict spoilage state for a single sensor sample dictionary."""
    df = pd.DataFrame([sample])
    pred = predict_dataframe(df)
    if pred is None:
        return compute_baseline_prediction(
            float(sample.get("gas_raw", 0)),
            float(sample.get("temperature_c", 25.0)),
            float(sample.get("humidity_percent", 50.0)),
        )
    return pred
