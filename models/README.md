# FreshGuard Machine Learning Models

This directory stores trained scikit-learn models serialized with `joblib`.

## Expected Model Format

To enable automatic ML spoilage classification on the dashboard:
1. Train a classifier on historical sensor records (`fresh`, `ripe`, `spoiled`).
2. Save the model as:
   ```text
   models/model.joblib
   ```

## Interface Requirements

The runtime inference hook in [`src/predictor.py`](../src/predictor.py) expects the model to accept a DataFrame with the following features:

```python
FEATURES = ["gas_raw", "temperature_c", "humidity_percent"]
```

And provide either or both:
- `model.predict(X)` -> returns predicted label string (e.g. `"fresh"`, `"ripe"`, `"spoiled"`)
- `model.predict_proba(X)` -> returns class probability distribution for confidence score calculation

If `models/model.joblib` is absent, the dashboard safely operates in hardware/telemetry mode with ML status indicated as `NOT TRAINED`.
