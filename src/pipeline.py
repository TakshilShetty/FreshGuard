"""Live End-to-End Data Pipeline for FreshGuard.

Streams sensor telemetry from ESP32 -> Ingests into Backend ->
Feeds to AI Engine (3-class classifier + shelf-life regressor + risk scoring) ->
Writes Raw & Processed Info to MongoDB & Memory Cache ->
Broadcasts via WebSocket -> Powers Server APIs and Streamlit Frontend.
"""
from __future__ import annotations

import argparse
import asyncio
import collections
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Set

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import serial
from pymongo import MongoClient

from src import ai_engine
from src.config import (
    DB_NAME,
    DEFAULT_BAUD,
    DEFAULT_SPECIMEN,
    MONGO_URI,
    READINGS_COLLECTION,
)
from src.sensor_reader import SOURCE_LABEL, StreamParser, list_ports

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("FreshGuard.Pipeline")

PROCESSED_COLLECTION = "processed_readings"

# Global set of WebSocket clients for live broadcasting
_ws_clients: Set[Any] = set()


def register_ws_client(ws) -> None:
    _ws_clients.add(ws)


def unregister_ws_client(ws) -> None:
    _ws_clients.discard(ws)


async def broadcast_to_clients(data: dict) -> None:
    """Send data to all connected WebSocket clients."""
    if not _ws_clients:
        return
    # Serialize once, send to all
    payload = json.dumps(data, default=str)
    dead = set()
    for ws in _ws_clients.copy():
        try:
            await ws.send_text(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _ws_clients.discard(ws)


def _sync_broadcast(data: dict) -> None:
    """Fire-and-forget broadcast from sync code."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(broadcast_to_clients(data))
        else:
            loop.run_until_complete(broadcast_to_clients(data))
    except RuntimeError:
        # No event loop available (pure CLI mode) — skip WS broadcast
        pass


class LivePipeline:
    """Manages continuous ingestion from serial, AI inference, and database dispatch."""

    def __init__(
        self,
        port: Optional[str] = None,
        baud: int = DEFAULT_BAUD,
        specimen_id: str = DEFAULT_SPECIMEN,
        mongo_uri: str = MONGO_URI,
        db_name: str = DB_NAME,
        cache_size: int = 200,
        on_reading: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.port = port
        self.baud = baud
        self.specimen_id = specimen_id
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.on_reading = on_reading
        self.running = False
        self.reading_count = 0

        # Ring buffer for high-throughput recent telemetry
        self.recent_readings: Deque[Dict[str, Any]] = collections.deque(maxlen=cache_size)

        # MongoDB client and collections
        self._mongo_client: Optional[MongoClient] = None
        self._readings_col = None
        self._processed_col = None
        self._init_mongo()

    def _init_mongo(self) -> None:
        try:
            self._mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=2500)
            db = self._mongo_client[self.db_name]
            self._readings_col = db[READINGS_COLLECTION]
            self._processed_col = db[PROCESSED_COLLECTION]
            # Verify connectivity
            self._mongo_client.admin.command("ping")
            logger.info("Connected to MongoDB at %s (DB: %s)", self.mongo_uri, self.db_name)
        except Exception as exc:
            logger.warning("MongoDB unavailable (%s). Running with in-memory pipeline cache only.", exc)
            self._readings_col = None
            self._processed_col = None

    def auto_detect_port(self) -> str:
        """Find the ESP32 COM port if not explicitly specified."""
        ports = list_ports()
        if not ports:
            raise RuntimeError("No serial ports found. Connect ESP32 via USB and install CP210x/CH340 drivers.")

        # Prioritize Silicon Labs CP210x, CH340, USB Serial
        for p in ports:
            dev = p.split()[0]
            desc = p.lower()
            if "cp210" in desc or "ch340" in desc or "esp32" in desc or "usb-to-uart" in desc:
                return dev

        # Fallback to first available COM port
        return ports[0].split()[0]

    def process_frame(self, raw_reading: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sensor reading: timestamp, validate, run AI engine, and store."""
        self.reading_count += 1
        now = datetime.now()
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")
        sample_id = f"{self.specimen_id}-{now:%Y%m%d%H%M%S}-{self.reading_count:04d}"

        # 1. Feed into AI Engine for inference (includes trend tracking)
        ai_result = ai_engine.predict_sample(raw_reading)

        # 2. Build enriched document matching FreshGuard schema
        document = {
            "timestamp": now,
            "sample_id": sample_id,
            "specimen_id": self.specimen_id,
            "gas_raw": raw_reading["gas_raw"],
            "temperature_c": raw_reading["temperature_c"],
            "humidity_percent": raw_reading["humidity_percent"],
            "source": SOURCE_LABEL,
            # AI predictions
            "spoilage_label": ai_result.get("label", "UNKNOWN"),
            "confidence": ai_result.get("confidence"),
            "probabilities": ai_result.get("probabilities", {}),
            "risk_score": ai_result.get("risk_score"),
            "risk_level": ai_result.get("risk_level"),
            "shelf_life_days": ai_result.get("shelf_life_days"),
            "shelf_life_hours": ai_result.get("shelf_life_hours"),
            "model_source": ai_result.get("model_source"),
            "alerts": ai_result.get("alerts", []),
            "processed": ai_result,
        }

        # 3. Cache in memory
        cached_copy = dict(document)
        cached_copy["timestamp_str"] = timestamp_str
        self.recent_readings.append(cached_copy)

        # 4. Dispatch to MongoDB collections
        if self._readings_col is not None:
            try:
                # Clean document for MongoDB (no nested complex types)
                mongo_doc = {k: v for k, v in document.items() if k != "processed"}
                mongo_doc["prediction"] = ai_result
                self._readings_col.insert_one(mongo_doc)

                if self._processed_col is not None:
                    self._processed_col.insert_one({
                        "sample_id": sample_id,
                        "timestamp": now,
                        "specimen_id": self.specimen_id,
                        "gas_raw": raw_reading["gas_raw"],
                        "temperature_c": raw_reading["temperature_c"],
                        "humidity_percent": raw_reading["humidity_percent"],
                        "prediction": ai_result,
                    })
            except Exception as exc:
                logger.warning("MongoDB write failed: %s", exc)

        # 5. Broadcast to WebSocket clients (non-blocking)
        ws_payload = {
            "type": "reading",
            "timestamp": timestamp_str,
            "specimen_id": self.specimen_id,
            "gas_raw": raw_reading["gas_raw"],
            "temperature_c": raw_reading["temperature_c"],
            "humidity_percent": raw_reading["humidity_percent"],
            "label": ai_result.get("label"),
            "confidence": ai_result.get("confidence"),
            "probabilities": ai_result.get("probabilities"),
            "risk_score": ai_result.get("risk_score"),
            "risk_level": ai_result.get("risk_level"),
            "shelf_life_days": ai_result.get("shelf_life_days"),
            "shelf_life_hours": ai_result.get("shelf_life_hours"),
            "alerts": ai_result.get("alerts", []),
            "trend": ai_result.get("trend"),
            "reading_number": self.reading_count,
        }
        _sync_broadcast(ws_payload)

        # 6. Notify external listener / callback if registered
        if self.on_reading:
            try:
                self.on_reading(document)
            except Exception as exc:
                logger.warning("on_reading callback raised: %s", exc)

        return document

    def get_latest_reading(self) -> Optional[Dict[str, Any]]:
        """Return the most recent processed reading from cache."""
        return self.recent_readings[-1] if self.recent_readings else None

    def get_recent_readings(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Return the recent N processed readings."""
        return list(self.recent_readings)[-limit:]

    def run(self) -> None:
        """Run the streaming pipeline loop. Auto-reconnects on disconnection."""
        if not self.port:
            self.port = self.auto_detect_port()

        self.running = True
        parser = StreamParser()

        model_file = ai_engine.get_model_file()
        bundle, _ = ai_engine.load_model_bundle()
        model_info = "Baseline Heuristic"
        if bundle and isinstance(bundle, dict):
            model_info = f"v{bundle.get('version', '?')} ({bundle.get('dataset_info', {}).get('total_readings', '?')} samples)"

        print(f"\n=======================================================")
        print(f" FreshGuard Live Sensor -> AI Engine -> Backend Pipeline")
        print(f"=======================================================")
        print(f" Port:          {self.port} (Baud: {self.baud})")
        print(f" Specimen:      {self.specimen_id}")
        print(f" Database:      {self.db_name}.{READINGS_COLLECTION}")
        print(f" AI Model:      {model_file or 'None'} — {model_info}")
        print(f" Capabilities:  3-class classification + Shelf-life estimation + Risk scoring")
        print(f" Status:        Streaming... (Press Ctrl+C to stop)\n")

        while self.running:
            try:
                with serial.Serial(self.port, self.baud, timeout=2.0) as ser:
                    ser.reset_input_buffer()
                    logger.info("Serial connection established on %s", self.port)

                    while self.running:
                        raw_line = ser.readline().decode("utf-8", errors="ignore")
                        if not raw_line:
                            continue

                        reading = parser.feed(raw_line)
                        if reading:
                            doc = self.process_frame(reading)
                            pred = doc.get("processed", {})
                            label = pred.get("label", "UNKNOWN")
                            conf = pred.get("confidence")
                            conf_str = f"{conf:.0%}" if conf is not None else "N/A"
                            risk = pred.get("risk_score", 0.0)
                            risk_lvl = pred.get("risk_level", "?")
                            shelf = pred.get("shelf_life_days")
                            shelf_str = f"{shelf:.1f}d" if shelf is not None else "N/A"

                            # Color-coded output
                            emoji = pred.get("risk_emoji", "")
                            print(
                                f"{emoji} [#{self.reading_count:04d}] "
                                f"Gas:{reading['gas_raw']:4d} | "
                                f"T:{reading['temperature_c']:4.1f}°C | "
                                f"H:{reading['humidity_percent']:4.1f}% | "
                                f"{label:9s} ({conf_str}) | "
                                f"Risk:{risk:4.0f}% {risk_lvl:8s} | "
                                f"Shelf:{shelf_str}"
                            )

                            # Print alerts
                            for alert in pred.get("alerts", []):
                                print(f"   {alert['icon']} [{alert['level']}] {alert['message']}")

            except serial.SerialException as s_err:
                logger.warning("Serial error on %s: %s. Reconnecting in 3s...", self.port, s_err)
                time.sleep(3.0)
            except KeyboardInterrupt:
                logger.info("Pipeline stopped by user (Ctrl+C). Total readings: %d", self.reading_count)
                self.running = False
                break
            except Exception as exc:
                logger.error("Unexpected error in pipeline: %s. Restarting loop...", exc)
                time.sleep(2.0)


# Singleton pipeline instance for server access
_pipeline_instance: Optional[LivePipeline] = None


def get_pipeline() -> Optional[LivePipeline]:
    return _pipeline_instance


def set_pipeline(pipeline: LivePipeline) -> None:
    global _pipeline_instance
    _pipeline_instance = pipeline


def main() -> None:
    ap = argparse.ArgumentParser(description="FreshGuard Live Data & AI Engine Pipeline")
    ap.add_argument("--port", help="Serial port of ESP32 (e.g. COM5)")
    ap.add_argument("--baud", type=int, default=DEFAULT_BAUD, help=f"Baud rate (default: {DEFAULT_BAUD})")
    ap.add_argument("--specimen", default=DEFAULT_SPECIMEN, help=f"Specimen ID (default: {DEFAULT_SPECIMEN})")
    args = ap.parse_args()

    pipeline = LivePipeline(port=args.port, baud=args.baud, specimen_id=args.specimen)
    set_pipeline(pipeline)
    pipeline.run()


if __name__ == "__main__":
    main()
