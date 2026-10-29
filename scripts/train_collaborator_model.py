#!/usr/bin/env python3
"""Train the FreshGuard AI model bundle from the combined banana dataset.

Trains:
1. RandomForestClassifier for 3-class banana spoilage classification
   (FRESH / RIPENING / SPOILED) with GroupKFold cross-validation
2. GradientBoostingRegressor for shelf-life estimation (remaining days)

Uses GroupShuffleSplit to prevent data leakage (whole bananas go to train or test).

Usage:
    python scripts/train_collaborator_model.py                   # default dataset
    python scripts/train_collaborator_model.py --data path.csv   # custom dataset
    python scripts/train_collaborator_model.py --excel path.xlsx  # from Excel
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

FEATURES = ["gas_raw", "temperature_c", "humidity_percent"]
LABEL = "spoilage_label"
LABELS = ["FRESH", "RIPENING", "SPOILED"]
MODEL_OUT = ROOT / "models" / "model.joblib"


def load_dataset(data_path: str | None = None, excel_path: str | None = None) -> pd.DataFrame:
    """Load dataset from CSV, Excel, or default combined file."""
    if excel_path:
        df = pd.read_excel(excel_path, sheet_name="Data")
    elif data_path:
        df = pd.read_csv(data_path)
    else:
        # Default: combined dataset
        combined = ROOT / "data" / "external" / "combined_banana_dataset.csv"
        if combined.exists():
            df = pd.read_csv(combined)
        else:
            # Fallback to Excel
            excel = ROOT / "banana_dataset.xlsx"
            if excel.exists():
                df = pd.read_excel(excel, sheet_name="Data")
            else:
                raise FileNotFoundError("No dataset found. Provide --data or --excel argument.")

    # Validate
    required = FEATURES + [LABEL, "sample_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df[LABEL] = df[LABEL].astype(str).str.strip().str.upper()
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)
    df = df[df[LABEL].isin(LABELS)].copy()

    return df


def train(df: pd.DataFrame) -> dict:
    """Train classifier + regressor and return model bundle."""
    X, y, groups = df[FEATURES], df[LABEL], df["sample_id"]

    print(f"\n{'='*60}")
    print(f" FreshGuard Model Training")
    print(f"{'='*60}")
    print(f" Dataset: {len(df)} readings from {groups.nunique()} bananas")
    print(f" Labels:  {dict(y.value_counts())}")
    print()

    # ========== CLASSIFIER ==========
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    tr, te = next(splitter.split(X, y, groups=groups))

    clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1, max_depth=12
    )
    clf.fit(X.iloc[tr], y.iloc[tr])
    pred = clf.predict(X.iloc[te])

    print("=== CLASSIFIER ===")
    print(f"Test Accuracy: {accuracy_score(y.iloc[te], pred):.4f}")
    print(classification_report(y.iloc[te], pred, labels=LABELS))

    # GroupKFold CV
    scores = []
    for a, b in GroupKFold(n_splits=5).split(X, y, groups=groups):
        m = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1, max_depth=12)
        m.fit(X.iloc[a], y.iloc[a])
        scores.append(accuracy_score(y.iloc[b], m.predict(X.iloc[b])))
    print(f"GroupKFold 5-fold: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

    # Final classifier on all data
    final_clf = RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1, max_depth=12
    )
    final_clf.fit(X, y)

    # ========== SHELF-LIFE REGRESSOR ==========
    shelf_rows = []
    for bid in df["sample_id"].unique():
        bdata = df[df["sample_id"] == bid].sort_values("day")
        spoiled_days = bdata[bdata[LABEL] == "SPOILED"]["day"]
        spoilage_day = spoiled_days.min() if len(spoiled_days) > 0 else 8
        for _, row in bdata.iterrows():
            remaining = max(0, spoilage_day - row["day"])
            shelf_rows.append({
                "gas_raw": row["gas_raw"],
                "temperature_c": row["temperature_c"],
                "humidity_percent": row["humidity_percent"],
                "sample_id": bid,
                "remaining_days": remaining,
            })

    shelf_df = pd.DataFrame(shelf_rows)
    X_shelf = shelf_df[FEATURES]
    y_shelf = shelf_df["remaining_days"]
    groups_shelf = shelf_df["sample_id"]

    mae_scores, rmse_scores = [], []
    for a, b in GroupKFold(n_splits=5).split(X_shelf, y_shelf, groups=groups_shelf):
        r = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42, learning_rate=0.1)
        r.fit(X_shelf.iloc[a], y_shelf.iloc[a])
        yp = r.predict(X_shelf.iloc[b])
        mae_scores.append(mean_absolute_error(y_shelf.iloc[b], yp))
        rmse_scores.append(np.sqrt(mean_squared_error(y_shelf.iloc[b], yp)))

    print(f"\n=== SHELF-LIFE REGRESSOR ===")
    print(f"GroupKFold MAE: {np.mean(mae_scores):.3f} ± {np.std(mae_scores):.3f} days")
    print(f"GroupKFold RMSE: {np.mean(rmse_scores):.3f} ± {np.std(rmse_scores):.3f} days")

    final_reg = GradientBoostingRegressor(n_estimators=300, max_depth=4, random_state=42, learning_rate=0.1)
    final_reg.fit(X_shelf, y_shelf)

    # Build bundle
    bundle = {
        "classifier": final_clf,
        "regressor": final_reg,
        "features": FEATURES,
        "classes": LABELS,
        "dataset_info": {
            "source": "combined (collaborator + research-augmented)",
            "total_samples": int(groups.nunique()),
            "total_readings": len(df),
            "days": int(df["day"].max()) + 1 if "day" in df.columns else 8,
            "classifier_cv_accuracy": round(float(np.mean(scores)), 4),
            "shelf_life_mae_days": round(float(np.mean(mae_scores)), 3),
        },
        "thresholds": {
            label: {
                "gas_min": int(df[df[LABEL] == label]["gas_raw"].min()),
                "gas_max": int(df[df[LABEL] == label]["gas_raw"].max()),
                "gas_mean": int(df[df[LABEL] == label]["gas_raw"].mean()),
            }
            for label in LABELS
        },
        "version": "2.0.0",
    }

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUT)
    print(f"\n✅ Model bundle saved to {MODEL_OUT}")
    print(f"   Classifier: {type(final_clf).__name__} ({final_clf.n_estimators} trees)")
    print(f"   Regressor: {type(final_reg).__name__}")
    print(f"   Feature importances: {dict(zip(FEATURES, final_clf.feature_importances_.round(4)))}")

    # Quick test predictions
    print("\n=== Quick Test ===")
    for gas, temp, hum, expected in [(400, 28, 60, "FRESH"), (900, 29, 65, "RIPENING"), (2000, 30, 70, "SPOILED")]:
        row = pd.DataFrame([{"gas_raw": gas, "temperature_c": temp, "humidity_percent": hum}])
        pred_label = final_clf.predict(row[FEATURES])[0]
        probs = dict(zip(final_clf.classes_, final_clf.predict_proba(row[FEATURES])[0]))
        shelf = final_reg.predict(row[FEATURES])[0]
        conf = max(probs.values())
        print(f"  gas={gas:4d} → {pred_label:9s} ({conf:.0%}) shelf={max(0, shelf):.1f}d [expected {expected}]")

    return bundle


def main():
    ap = argparse.ArgumentParser(description="Train FreshGuard AI Model")
    ap.add_argument("--data", help="Path to CSV dataset")
    ap.add_argument("--excel", help="Path to Excel dataset (reads 'Data' sheet)")
    args = ap.parse_args()

    df = load_dataset(args.data, args.excel)
    train(df)


if __name__ == "__main__":
    main()
