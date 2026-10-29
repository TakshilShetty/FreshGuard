from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATA_FILE = Path("banana_data.csv")
MODEL_FILE = Path("banana_rf_model.joblib")

# All three sensor inputs
FEATURES = [
    "gas_raw",           # MQ-3
    "temperature_c",     # DHT22
    "humidity_percent",  # DHT22
]

LABEL = "spoilage_label"
ALLOWED_LABELS = {"FRESH", "RIPENING", "SPOILED"}


def train_model():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    df = pd.read_csv(DATA_FILE)

    required = FEATURES + [LABEL, "sample_id"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Missing CSV columns: {missing}")

    # Normalize labels and convert sensor values to numbers.
    df[LABEL] = df[LABEL].astype(str).str.strip().str.upper()

    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove incomplete or invalid rows.
    df = df.dropna(subset=required)
    df = df[df[LABEL].isin(ALLOWED_LABELS)].copy()

    if df.empty or df[LABEL].nunique() < 2:
        raise ValueError(
            "Need valid readings with at least two freshness classes."
        )

    # Each banana sample must belong to only one split.
    if df["sample_id"].nunique() < 2:
        raise ValueError(
            "Collect readings from multiple banana samples "
            "before evaluating the model."
        )

    X = df[FEATURES]
    y = df[LABEL]
    groups = df["sample_id"]

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.25,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    if y_train.nunique() < 2:
        raise ValueError(
            "Training split has fewer than two classes. "
            "Collect more banana samples across the classes."
        )

    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("\n--- MODEL EVALUATION ---")
    print("Accuracy:", round(
        accuracy_score(y_test, predictions), 3
    ))

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=sorted(ALLOWED_LABELS),
        zero_division=0,
    ))

    print("Confusion matrix (alphabetical label order):")
    print(sorted(ALLOWED_LABELS))
    print(confusion_matrix(
        y_test,
        predictions,
        labels=sorted(ALLOWED_LABELS),
    ))

    # Save the trained model and feature order.
    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_FILE,
    )

    print(f"\nModel saved to: {MODEL_FILE.resolve()}")

    print("\nFeature importance:")
    importance = pd.Series(
        model.feature_importances_,
        index=FEATURES,
    ).sort_values(ascending=False)

    print(importance)


def predict_freshness(gas_raw, temperature_c, humidity_percent):
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model() first."
        )

    saved = joblib.load(MODEL_FILE)
    model = saved["model"]
    features = saved["features"]

    # Build one row using the exact training feature order.
    readings = pd.DataFrame([{
        "gas_raw": gas_raw,
        "temperature_c": temperature_c,
        "humidity_percent": humidity_percent,
    }])[features]

    readings = readings.apply(pd.to_numeric, errors="coerce")

    if readings.isna().any().any():
        raise ValueError("Sensor readings must be valid numbers.")

    prediction = model.predict(readings)[0]
    probabilities = model.predict_proba(readings)[0]

    print("\n--- LIVE PREDICTION ---")
    print("Predicted banana status:", prediction)
    print("Class probabilities:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label}: {probability:.1%}")

    return prediction


if __name__ == "__main__":
    # Train and evaluate using the labelled CSV.
    train_model()

    # After training, call predict_freshness() whenever
    # your ESP32 software supplies a new sensor reading.
    #
    # Example values ONLY:
    #
    # predict_freshness(
    #     gas_raw=1700,
    #     temperature_c=28.0,
    #     humidity_percent=68.0,
    # )
