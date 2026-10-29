"""Train and export a collaborator machine learning model for SporeWatch / FreshGuard.

Usage:
    python scripts/train_collaborator_model.py
    python scripts/train_collaborator_model.py --data data/demo/demo_5day_3bananas.csv --output models/model.joblib

This script provides an out-of-the-box working ML baseline for collaborators.
Collaborators can customize the algorithm, feature engineering, hyperparameters, or training data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.ai_engine import STANDARD_FEATURES


def train_model(data_path: Path, output_path: Path, n_estimators: int = 100) -> None:
    if not data_path.is_file():
        sys.exit(f"Dataset not found: {data_path}")

    print(f"Loading training data from {data_path}...")
    df = pd.read_csv(data_path)

    for col in STANDARD_FEATURES + ["spoilage_label"]:
        if col not in df.columns:
            sys.exit(f"Required column '{col}' missing from {data_path}")

    X = df[STANDARD_FEATURES]
    y = df["spoilage_label"].astype(str)

    print(f"Dataset shape: {X.shape}. Target class distribution:\n{y.value_counts()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining RandomForestClassifier (n_estimators={n_estimators})...")
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("\nModel Evaluation Report:")
    print(classification_report(y_test, y_pred))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, output_path)
    print(f"Successfully exported trained model to: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Train collaborator spoilage classifier model")
    ap.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "demo" / "demo_5day_3bananas.csv",
        help="Path to labeled training CSV dataset",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "models" / "model.joblib",
        help="Destination path for serialized model",
    )
    ap.add_argument("--trees", type=int, default=100, help="Number of trees (default: 100)")
    args = ap.parse_args()

    train_model(args.data, args.output, n_estimators=args.trees)


if __name__ == "__main__":
    main()
