# FreshGuard Machine Learning & AI Engine

This directory stores trained machine learning model bundles for the FreshGuard banana freshness, spoilage risk, and shelf-life monitoring pipeline.

## Model Bundle v2.0.0 Architecture

The active model at `models/model.joblib` is a bundled container supporting multi-task inference:

```python
{
    "classifier": RandomForestClassifier(...),       # 3-class: FRESH, RIPENING, SPOILED
    "regressor": GradientBoostingRegressor(...),     # Shelf-life in remaining days
    "features": ["gas_raw", "temperature_c", "humidity_percent"],
    "classes": ["FRESH", "RIPENING", "SPOILED"],
    "thresholds": {...},                             # Per-class empirical ranges
    "dataset_info": {...},                           # Training metadata & GroupKFold CV scores
    "version": "2.0.0"
}
```

The AI Engine (`src/ai_engine.py`) continuously tracks the file's modification timestamp and **automatically hot-reloads** any newly saved model on the fly without stopping backend services.

---

## Capabilities & Outputs

When a sensor reading arrives from the ESP32 (MQ-3 + DHT22), the engine produces:

| Output Field | Type | Description |
| :--- | :--- | :--- |
| `label` | `str` | `FRESH`, `RIPENING`, or `SPOILED` |
| `confidence` | `float` | Maximum class probability (0.00 – 1.00) |
| `probabilities` | `dict` | Probabilities for `{FRESH, RIPENING, SPOILED}` |
| `risk_score` | `float` | Composite spoilage risk score (`0.0` to `100.0%`) |
| `risk_level` | `str` | `LOW`, `MEDIUM`, `HIGH`, or `CRITICAL` |
| `shelf_life_days` | `float` | Estimated remaining shelf life in days |
| `shelf_life_hours` | `float` | Estimated remaining shelf life in hours |
| `trend` | `dict` | Sliding-window gas slope, rate, and deterioration speed |
| `alerts` | `list` | Early-warning contextual alerts & threshold warnings |

---

## Training a Model

### Using the Combined Dataset (Collaborator + Research-Augmented):
```bash
python scripts/train_collaborator_model.py
```

### Using Custom CSV or Excel:
```bash
python scripts/train_collaborator_model.py --excel data/banana_dataset.xlsx
python scripts/train_collaborator_model.py --data data/external/combined_banana_dataset.csv
```

---

## Testing Your Model

### Via Python:
```python
from src.ai_engine import predict_sample

result = predict_sample({"gas_raw": 1950, "temperature_c": 28.5, "humidity_percent": 65.0})
print(result["label"])            # 'SPOILED'
print(result["confidence"])       # 0.99
print(result["shelf_life_days"])  # 0.0
print(result["risk_level"])       # 'CRITICAL'
print(result["alerts"])           # [{'level': 'CRITICAL', ...}]
```

### Via REST API:
```bash
curl -X POST http://localhost:8000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"gas_raw": 1950, "temperature_c": 28.5, "humidity_percent": 65.0}'
```

### Real-Time WebSocket:
Connect to `ws://localhost:8000/ws/live` to receive live JSON frames streamed as each ESP32 reading is processed.
