"""AI hook. predict(df) -> {"label": str, "confidence": float} or None when no trained model exists.

Drop a joblib model at models/model.joblib exposing predict(X)/predict_proba(X) over
[gas_raw, temperature_c, humidity_percent] (latest row), or replace this function.
"""
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.joblib"
FEATURES = ["gas_raw", "temperature_c", "humidity_percent"]


def predict(df):
    if not MODEL_PATH.exists() or df.empty:
        return None
    import joblib  # lazy: only needed once a model exists

    model = joblib.load(MODEL_PATH)
    x = df[FEATURES].tail(1)
    proba = model.predict_proba(x)[0] if hasattr(model, "predict_proba") else None
    return {"label": str(model.predict(x)[0]), "confidence": float(proba.max()) if proba is not None else None}
