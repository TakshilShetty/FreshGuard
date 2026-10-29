# FreshGuard Machine Learning & AI Engine

This directory stores trained machine learning models for the FreshGuard / SporeWatch banana freshness monitoring pipeline.

## Collaborator Quickstart

Collaborators working on the AI engine can build, evaluate, and deploy models seamlessly into the live data pipeline without stopping or restarting the backend servers.

### 1. Model Location & Formats
Save your trained model file to:
```text
models/model.joblib   (Primary format)
# or
models/model.pkl      (Secondary pickle format)
```
The AI Engine (`src/ai_engine.py`) continuously tracks the file's modification timestamp and **automatically hot-reloads** your updated model on the fly.

---

### 2. Expected Inputs & Features
The model must accept a 2D array or pandas DataFrame with the following standardized feature columns:

| Feature Name | Type | Description | Typical Range |
| :--- | :--- | :--- | :--- |
| `gas_raw` | `float` / `int` | MQ-3 sensor 12-bit ADC reading | `0` to `4095` |
| `temperature_c` | `float` | Ambient temperature in Celsius | `15.0` to `45.0` |
| `humidity_percent` | `float` | Relative humidity percentage | `20.0` to `95.0` |

---

### 3. Model Interface Requirements
Your serialized object should expose standard scikit-learn compatible methods:
- `model.predict(X)`: Returns an array or list of predicted label strings (e.g. `["fresh"]`, `["ripe"]`, `["spoiled"]`, `["no_banana"]`).
- `model.predict_proba(X)` *(optional but recommended)*: Returns probability distributions across `model.classes_`, enabling the dashboard to display real-time confidence scores.

---

### 4. Training a Model
You can use the built-in reference training pipeline:
```bash
python scripts/train_collaborator_model.py
```
Or customize parameters:
```bash
python scripts/train_collaborator_model.py --data data/demo/demo_5day_3bananas.csv --output models/model.joblib --trees 150
```

---

### 5. Testing Your Model

#### Via Python:
```python
from src.ai_engine import predict_sample

result = predict_sample({"gas_raw": 1050, "temperature_c": 28.5, "humidity_percent": 55.0})
print(result)
# Output: {'label': 'fresh', 'confidence': 0.88, 'risk_score': 15.0, 'model_source': 'collaborator_model', ...}
```

#### Via Backend REST API:
```bash
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"gas_raw": 1050, "temperature_c": 28.5, "humidity_percent": 55.0}'
```

#### Via Live Hardware Pipeline:
When the ESP32 is running, start the pipeline:
```bash
python scripts/ingest.py
```
Readings from the physical sensors will be fed directly into your model in real time and displayed on the Streamlit dashboard at `http://localhost:5000`.
