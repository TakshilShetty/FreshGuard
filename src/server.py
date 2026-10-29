"""FreshGuard Backend API Server with WebSocket real-time streaming.

Provides RESTful HTTP endpoints + WebSocket for live sensor telemetry,
AI Engine inference (3-class + shelf-life + risk), and collaborator tools.

Run with:
    python -m src.server
    uvicorn src.server:app --port 8000 --reload
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient

from src import ai_engine
from src.config import DB_NAME, MONGO_URI, READINGS_COLLECTION
from src.pipeline import register_ws_client, unregister_ws_client
from src.sensor_reader import list_ports

app = FastAPI(
    title="FreshGuard Sensor & AI Engine API",
    description=(
        "Backend service integrating ESP32 telemetry, 3-class banana spoilage classifier, "
        "shelf-life estimation, risk scoring, and real-time WebSocket streaming."
    ),
    version="2.0.0",
)

# Enable CORS for frontend and collaborator clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SensorReadingInput(BaseModel):
    gas_raw: float = Field(..., ge=0, le=4095, description="MQ-3 raw ADC count (0-4095)")
    temperature_c: float = Field(..., description="Ambient temperature in degrees Celsius")
    humidity_percent: float = Field(..., ge=0, le=100, description="Relative humidity percentage (0-100)")


def get_mongo_db():
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=1500)
        return client[DB_NAME]
    except Exception:
        return None


@app.get("/")
def root():
    return {
        "service": "FreshGuard Sensor & AI Engine API",
        "version": "2.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "capabilities": [
            "3-class classification (FRESH/RIPENING/SPOILED)",
            "Shelf-life estimation (days/hours)",
            "Risk scoring (0-100, LOW/MEDIUM/HIGH/CRITICAL)",
            "Trend analysis & early-warning alerts",
            "Real-time WebSocket streaming",
        ],
        "endpoints": {
            "status": "/api/status",
            "latest_reading": "/api/readings/latest",
            "history": "/api/readings/history",
            "predict": "/api/predict",
            "reload_model": "/api/model/reload",
            "model_info": "/api/model/info",
            "trend": "/api/trend",
            "websocket": "ws://localhost:8000/ws/live",
        },
    }


@app.get("/api/status")
def get_system_status():
    db = get_mongo_db()
    mongo_ok = False
    doc_count = 0
    if db is not None:
        try:
            doc_count = db[READINGS_COLLECTION].count_documents({})
            mongo_ok = True
        except Exception:
            mongo_ok = False

    bundle, path = ai_engine.load_model_bundle()
    classifier, regressor = ai_engine._get_classifier_and_regressor(bundle)

    model_info = {
        "available": classifier is not None,
        "source": "trained_model" if classifier is not None else "baseline_heuristic",
        "file": path.name if path else None,
        "version": bundle.get("version", "unknown") if isinstance(bundle, dict) else None,
        "has_classifier": classifier is not None,
        "has_regressor": regressor is not None,
        "classes": [str(c) for c in getattr(classifier, "classes_", [])] if classifier else ["FRESH", "RIPENING", "SPOILED"],
        "dataset_info": bundle.get("dataset_info") if isinstance(bundle, dict) else None,
    }

    ports = list_ports()

    return {
        "status": "healthy",
        "hardware": {
            "available_ports": ports,
            "connected": len(ports) > 0,
        },
        "database": {
            "connected": mongo_ok,
            "database": DB_NAME,
            "readings_collection": READINGS_COLLECTION,
            "total_documents": doc_count,
        },
        "ai_engine": model_info,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/readings/latest")
def get_latest_reading(specimen: Optional[str] = None):
    db = get_mongo_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB database connection unavailable")

    query = {"specimen_id": specimen} if specimen else {}
    doc = db[READINGS_COLLECTION].find_one(query, {"_id": 0}, sort=[("timestamp", -1)])
    if not doc:
        raise HTTPException(status_code=404, detail="No readings found in database")

    if isinstance(doc.get("timestamp"), datetime):
        doc["timestamp"] = doc["timestamp"].isoformat()

    return doc


@app.get("/api/readings/history")
def get_readings_history(limit: int = 50, specimen: Optional[str] = None):
    db = get_mongo_db()
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB database connection unavailable")

    limit = max(1, min(limit, 500))
    query = {"specimen_id": specimen} if specimen else {}
    docs = list(db[READINGS_COLLECTION].find(query, {"_id": 0}).sort("timestamp", -1).limit(limit))[::-1]

    for d in docs:
        if isinstance(d.get("timestamp"), datetime):
            d["timestamp"] = d["timestamp"].isoformat()

    return {"specimen": specimen or "all", "count": len(docs), "readings": docs}


@app.post("/api/predict")
def predict_sample(reading: SensorReadingInput):
    """Run the full AI pipeline on a single sensor reading.

    Returns classification (FRESH/RIPENING/SPOILED), confidence, risk score,
    shelf-life estimate, and any alerts.
    """
    sample_dict = {
        "gas_raw": reading.gas_raw,
        "temperature_c": reading.temperature_c,
        "humidity_percent": reading.humidity_percent,
    }
    prediction = ai_engine.predict_sample(sample_dict)
    return {
        "input": sample_dict,
        "prediction": prediction,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/api/model/reload")
def reload_model():
    """Force the AI Engine to check and reload model files from models/."""
    # Reset cached model to force reload
    ai_engine._CACHED_BUNDLE = None
    ai_engine._CACHED_MTIME = 0.0
    bundle, path = ai_engine.load_model_bundle()
    classifier, regressor = ai_engine._get_classifier_and_regressor(bundle)
    return {
        "reloaded": True,
        "model_file": path.name if path else None,
        "classifier_loaded": classifier is not None,
        "regressor_loaded": regressor is not None,
        "version": bundle.get("version") if isinstance(bundle, dict) else None,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/model/info")
def model_info():
    """Detailed information about the loaded AI model."""
    bundle, path = ai_engine.load_model_bundle()
    if bundle is None:
        return {"loaded": False, "info": "No model file found in models/ directory."}

    classifier, regressor = ai_engine._get_classifier_and_regressor(bundle)
    info = {
        "loaded": True,
        "file": path.name if path else None,
        "version": bundle.get("version") if isinstance(bundle, dict) else "legacy",
        "classifier": {
            "type": type(classifier).__name__ if classifier else None,
            "classes": [str(c) for c in getattr(classifier, "classes_", [])],
            "n_estimators": getattr(classifier, "n_estimators", None),
            "feature_importances": dict(zip(
                bundle.get("features", ai_engine.STANDARD_FEATURES),
                [round(float(x), 4) for x in getattr(classifier, "feature_importances_", [])]
            )) if classifier else None,
        },
        "regressor": {
            "type": type(regressor).__name__ if regressor else None,
            "available": regressor is not None,
        },
        "features": bundle.get("features") if isinstance(bundle, dict) else ai_engine.STANDARD_FEATURES,
        "dataset_info": bundle.get("dataset_info") if isinstance(bundle, dict) else None,
        "thresholds": bundle.get("thresholds") if isinstance(bundle, dict) else None,
    }
    return info


@app.get("/api/trend")
def get_trend():
    """Current trend analysis from the sliding window."""
    tracker = ai_engine.get_trend_tracker()
    trend = tracker.get_trend()
    return {"trend": trend, "timestamp": datetime.now().isoformat()}


# ---------------------------------------------------------------------------
# WebSocket endpoint for real-time streaming
# ---------------------------------------------------------------------------
@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Real-time WebSocket feed of processed sensor readings.

    Clients receive JSON messages with classification, risk, shelf-life, and alerts
    as each ESP32 reading is processed by the pipeline.
    """
    await websocket.accept()
    register_ws_client(websocket)
    try:
        while True:
            # Keep connection alive; the pipeline broadcasts data to us
            data = await websocket.receive_text()
            # Client can send "ping" to keep alive
            if data.strip().lower() == "ping":
                await websocket.send_text('{"type": "pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        unregister_ws_client(websocket)


def main():
    import uvicorn

    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
