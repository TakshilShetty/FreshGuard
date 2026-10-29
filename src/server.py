"""FreshGuard Backend API Server.

Provides RESTful HTTP endpoints for live sensor telemetry, AI Engine inference,
MongoDB status, and collaborator model evaluation.

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

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pymongo import MongoClient

from src import ai_engine
from src.config import DB_NAME, MONGO_URI, READINGS_COLLECTION
from src.sensor_reader import list_ports

app = FastAPI(
    title="FreshGuard Sensor & AI Engine API",
    description="Backend service integrating ESP32 telemetry, Collaborator AI Engine, and MongoDB",
    version="1.0.0",
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
        "version": "1.0.0",
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "status": "/api/status",
            "latest_reading": "/api/readings/latest",
            "history": "/api/readings/history",
            "predict": "/api/predict",
            "reload_model": "/api/model/reload",
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

    model, path = ai_engine.load_collaborator_model()
    model_info = {
        "available": model is not None,
        "source": "collaborator_joblib" if model is not None else "baseline_heuristic",
        "file": path.name if path else None,
        "classes": [str(c) for c in getattr(model, "classes_", [])] if model is not None else ["fresh", "ripe", "spoiled"],
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
    """Collaborator endpoint to evaluate sensor readings against the AI Engine."""
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
    """Force the AI Engine to check and reload collaborator model files from models/."""
    model, path = ai_engine.load_collaborator_model()
    return {
        "reloaded": True,
        "model_file": path.name if path else None,
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
    }


def main():
    import uvicorn

    uvicorn.run("src.server:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
