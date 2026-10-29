"""FreshGuard AI Engine — 3-class classification, shelf-life estimation, risk scoring, and trend analysis.

Handles:
- Model loading with hot-reload (watches models/model.joblib mtime)
- 3-class classification: FRESH / RIPENING / SPOILED with per-class probabilities
- Shelf-life estimation in days (GradientBoostingRegressor)
- Risk score (0-100) with LOW / MEDIUM / HIGH / CRITICAL levels
- Trend analysis over a sliding window of recent readings
- Confidence calibration and early-warning alerts
- Fallback heuristic when no trained model is available

The model bundle (models/model.joblib) contains:
  - 'classifier': RandomForestClassifier (FRESH/RIPENING/SPOILED)
  - 'regressor': GradientBoostingRegressor (remaining shelf-life in days)
  - 'features': ['gas_raw', 'temperature_c', 'humidity_percent']
  - 'classes': ['FRESH', 'RIPENING', 'SPOILED']
  - 'thresholds': per-class gas ranges from training data
"""
from __future__ import annotations

import logging
import time
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
PRIMARY_MODEL_PATH = MODELS_DIR / "model.joblib"
SECONDARY_MODEL_PATH = MODELS_DIR / "model.pkl"
STANDARD_FEATURES = ["gas_raw", "temperature_c", "humidity_percent"]

# Labels the classifier can output
LABELS = ["FRESH", "RIPENING", "SPOILED"]

# Risk level thresholds (risk_score -> level)
RISK_LEVELS = [
    (0, 25, "LOW", "🟢"),
    (25, 55, "MEDIUM", "🟡"),
    (55, 80, "HIGH", "🟠"),
    (80, 101, "CRITICAL", "🔴"),
]

# In-memory cached model and its last modification timestamp
_CACHED_BUNDLE: Any = None
_CACHED_MTIME: float = 0.0
_CACHED_PATH: Optional[Path] = None


# ---------------------------------------------------------------------------
# Trend tracker — sliding window for real-time analysis
# ---------------------------------------------------------------------------
class TrendTracker:
    """Maintains a sliding window of recent sensor readings for trend analysis."""

    def __init__(self, window_size: int = 30) -> None:
        self.window_size = window_size
        self.readings: Deque[Dict[str, Any]] = deque(maxlen=window_size)

    def add(self, reading: Dict[str, Any]) -> None:
        self.readings.append({
            "gas_raw": reading["gas_raw"],
            "temperature_c": reading["temperature_c"],
            "humidity_percent": reading["humidity_percent"],
            "timestamp": reading.get("timestamp", time.time()),
        })

    def get_trend(self) -> Dict[str, Any]:
        """Compute trend statistics over the sliding window."""
        if len(self.readings) < 2:
            return {
                "samples": len(self.readings),
                "gas_trend": "INSUFFICIENT_DATA",
                "gas_rate": 0.0,
                "gas_avg": self.readings[-1]["gas_raw"] if self.readings else 0,
                "gas_min": self.readings[-1]["gas_raw"] if self.readings else 0,
                "gas_max": self.readings[-1]["gas_raw"] if self.readings else 0,
                "temp_avg": self.readings[-1]["temperature_c"] if self.readings else 0,
                "humidity_avg": self.readings[-1]["humidity_percent"] if self.readings else 0,
                "deterioration_speed": "UNKNOWN",
            }

        gas_values = [r["gas_raw"] for r in self.readings]
        temps = [r["temperature_c"] for r in self.readings]
        humidities = [r["humidity_percent"] for r in self.readings]

        n = len(gas_values)
        gas_arr = np.array(gas_values, dtype=float)

        # Linear regression for gas trend (slope)
        x = np.arange(n, dtype=float)
        slope = np.polyfit(x, gas_arr, 1)[0] if n >= 2 else 0.0

        # Rate of change per reading
        gas_rate = slope

        # Determine trend direction
        if abs(slope) < 2.0:
            gas_trend = "STABLE"
        elif slope > 0:
            gas_trend = "RISING" if slope < 15 else "RISING_FAST"
        else:
            gas_trend = "FALLING" if slope > -15 else "FALLING_FAST"

        # Deterioration speed based on gas acceleration
        recent_half = gas_arr[n // 2:]
        early_half = gas_arr[: n // 2]
        accel = float(recent_half.mean() - early_half.mean()) if len(early_half) > 0 else 0

        if accel > 100:
            deterioration = "ACCELERATING"
        elif accel > 30:
            deterioration = "MODERATE"
        elif accel > 5:
            deterioration = "SLOW"
        elif abs(accel) <= 5:
            deterioration = "STABLE"
        else:
            deterioration = "IMPROVING"

        return {
            "samples": n,
            "gas_trend": gas_trend,
            "gas_rate": round(float(gas_rate), 2),
            "gas_avg": round(float(gas_arr.mean()), 1),
            "gas_min": int(gas_arr.min()),
            "gas_max": int(gas_arr.max()),
            "gas_std": round(float(gas_arr.std()), 1),
            "temp_avg": round(float(np.mean(temps)), 1),
            "temp_min": round(float(np.min(temps)), 1),
            "temp_max": round(float(np.max(temps)), 1),
            "humidity_avg": round(float(np.mean(humidities)), 1),
            "deterioration_speed": deterioration,
        }


# Global trend tracker instance
_trend_tracker = TrendTracker(window_size=60)


def get_trend_tracker() -> TrendTracker:
    return _trend_tracker


# ---------------------------------------------------------------------------
# Transit tracker — MPU-6050 physical handling, shock, tilt, and road stress
# ---------------------------------------------------------------------------
class TransitTracker:
    """Tracks physical transit telemetry: shocks, drops, orientation, vibrations, and handling stress."""

    def __init__(self, window_size: int = 40) -> None:
        self.window_size = window_size
        self.readings: Deque[Dict[str, Any]] = deque(maxlen=window_size)
        self.peak_shock_g: float = 1.0
        self.severe_drop_count: int = 0
        self.mild_bump_count: int = 0
        self.tilt_violation_count: int = 0
        self.incident_log: Deque[Dict[str, Any]] = deque(maxlen=20)
        self._last_incident_time: float = 0.0

    def add(self, reading: Dict[str, Any]) -> None:
        raw_ts = reading.get("timestamp", time.time())
        if isinstance(raw_ts, (int, float)):
            ts_str = time.strftime("%H:%M:%S", time.localtime(raw_ts))
            now_sec = float(raw_ts)
        else:
            ts_str = str(raw_ts).split()[-1] if " " in str(raw_ts) else str(raw_ts)
            now_sec = time.time()

        # Acceleration in g's
        ax = float(reading.get("accel_x", 0.0))
        ay = float(reading.get("accel_y", 0.0))
        az = float(reading.get("accel_z", 1.0))

        # Instantaneous total shock magnitude
        if "shock_g" in reading and float(reading["shock_g"]) > 0:
            shock = float(reading["shock_g"])
        else:
            shock = float(np.sqrt(ax * ax + ay * ay + az * az))
        if not np.isfinite(shock) or shock <= 0:
            shock = 1.0

        # Pitch and Roll in degrees
        if "pitch_deg" in reading:
            pitch = float(reading["pitch_deg"])
        else:
            pitch = float(np.degrees(np.arctan2(-ax, np.sqrt(ay * ay + az * az))))

        if "roll_deg" in reading:
            roll = float(reading["roll_deg"])
        else:
            roll = float(np.degrees(np.arctan2(ay, az)))

        if shock > self.peak_shock_g:
            self.peak_shock_g = round(shock, 2)

        # Incident logging with 1.5s rate-limit
        can_log = (now_sec - self._last_incident_time) >= 1.5

        if shock >= 2.5:
            self.severe_drop_count += 1
            if can_log:
                self.incident_log.append({
                    "timestamp": ts_str,
                    "type": "SEVERE_DROP",
                    "severity": f"{shock:.1f}g",
                    "level": "CRITICAL",
                    "icon": "💥",
                    "message": f"High impact drop ({shock:.1f}g). Severe bruising risk.",
                })
                self._last_incident_time = now_sec
        elif shock >= 1.6:
            self.mild_bump_count += 1
            if can_log and len(self.incident_log) < 20:
                self.incident_log.append({
                    "timestamp": ts_str,
                    "type": "MILD_BUMP",
                    "severity": f"{shock:.1f}g",
                    "level": "CAUTION",
                    "icon": "⚠️",
                    "message": f"Transit bump detected ({shock:.1f}g).",
                })
                self._last_incident_time = now_sec

        # Tilt violations (>35 degrees)
        is_tilted = abs(pitch) > 35.0 or abs(roll) > 35.0
        if is_tilted:
            self.tilt_violation_count += 1
            if can_log and (now_sec - self._last_incident_time) >= 4.0:
                self.incident_log.append({
                    "timestamp": ts_str,
                    "type": "TILT_VIOLATION",
                    "severity": f"P:{pitch:.0f}° R:{roll:.0f}°",
                    "level": "WARNING",
                    "icon": "📦",
                    "message": f"Crate tilted (Pitch: {pitch:.0f}°, Roll: {roll:.0f}°). Risk of crush injury.",
                })
                self._last_incident_time = now_sec

        self.readings.append({
            "accel_x": ax, "accel_y": ay, "accel_z": az,
            "shock_g": shock, "pitch_deg": pitch, "roll_deg": roll,
            "timestamp": ts_str,
        })

    def get_transit_state(self) -> Dict[str, Any]:
        """Compute transit indicators, handling integrity, and bruise penalty."""
        if not self.readings:
            return {
                "live_shock_g": 1.0,
                "peak_shock_g": 1.0,
                "accel_x": 0.0, "accel_y": 0.0, "accel_z": 1.0,
                "pitch_deg": 0.0, "roll_deg": 0.0,
                "tilt_status": "LEVEL & UPRIGHT",
                "road_condition": "STATIONARY",
                "vibration_rms": 0.0,
                "severe_drops": 0,
                "mild_bumps": 0,
                "tilt_violations": 0,
                "handling_integrity_score": 100.0,
                "bruise_penalty_percent": 0.0,
                "incident_log": [],
            }

        latest = self.readings[-1]
        shocks = [r["shock_g"] for r in self.readings]

        # Vibration RMS: deviation from 1.0g
        devs = [s - 1.0 for s in shocks]
        rms = float(np.sqrt(np.mean(np.square(devs))))

        if rms < 0.12:
            road = "STATIONARY"
        elif rms < 0.45:
            road = "SMOOTH_HIGHWAY"
        else:
            road = "ROUGH_ROAD"

        pitch = latest["pitch_deg"]
        roll = latest["roll_deg"]
        az = latest["accel_z"]

        if az < -0.3:
            tilt = "INVERTED (UPSIDE DOWN)"
        elif abs(pitch) > 45 or abs(roll) > 45:
            tilt = "CRITICAL TILT / TIPPING"
        elif abs(pitch) > 25 or abs(roll) > 25:
            tilt = "MODERATE TILT"
        else:
            tilt = "LEVEL & UPRIGHT"

        # Physical Handling Integrity (starts at 100)
        drop_loss = self.severe_drop_count * 15.0
        bump_loss = min(20.0, self.mild_bump_count * 2.0)
        tilt_loss = min(15.0, (self.tilt_violation_count // 5) * 3.0)
        rough_loss = 10.0 if road == "ROUGH_ROAD" else 0.0

        integrity = max(0.0, 100.0 - (drop_loss + bump_loss + tilt_loss + rough_loss))

        # Bruise penalty on shelf life
        penalty = min(0.70, (self.severe_drop_count * 0.15) + (self.mild_bump_count * 0.02) + (0.05 if road == "ROUGH_ROAD" else 0.0))

        return {
            "live_shock_g": round(latest["shock_g"], 2),
            "peak_shock_g": round(self.peak_shock_g, 2),
            "accel_x": round(latest["accel_x"], 2),
            "accel_y": round(latest["accel_y"], 2),
            "accel_z": round(latest["accel_z"], 2),
            "pitch_deg": round(pitch, 1),
            "roll_deg": round(roll, 1),
            "tilt_status": tilt,
            "road_condition": road,
            "vibration_rms": round(rms, 3),
            "severe_drops": self.severe_drop_count,
            "mild_bumps": self.mild_bump_count,
            "tilt_violations": self.tilt_violation_count,
            "handling_integrity_score": round(integrity, 1),
            "bruise_penalty_percent": round(penalty * 100.0, 1),
            "incident_log": list(self.incident_log)[::-1],  # newest first
        }


# Global transit tracker instance
_transit_tracker = TransitTracker(window_size=60)


def get_transit_tracker() -> TransitTracker:
    return _transit_tracker


# ---------------------------------------------------------------------------
# Model loading with hot-reload
# ---------------------------------------------------------------------------
def get_model_file() -> Optional[Path]:
    """Return the active model path if one exists, prioritizing model.joblib then model.pkl."""
    if PRIMARY_MODEL_PATH.is_file():
        return PRIMARY_MODEL_PATH
    if SECONDARY_MODEL_PATH.is_file():
        return SECONDARY_MODEL_PATH
    return None


def load_model_bundle() -> Tuple[Optional[Any], Optional[Path]]:
    """Hot-reload model bundle if modified or newly created."""
    global _CACHED_BUNDLE, _CACHED_MTIME, _CACHED_PATH

    model_file = get_model_file()
    if model_file is None:
        _CACHED_BUNDLE = None
        _CACHED_MTIME = 0.0
        _CACHED_PATH = None
        return None, None

    try:
        mtime = model_file.stat().st_mtime
        if _CACHED_BUNDLE is None or _CACHED_PATH != model_file or mtime > _CACHED_MTIME:
            import joblib

            _CACHED_BUNDLE = joblib.load(model_file)
            _CACHED_MTIME = mtime
            _CACHED_PATH = model_file
            logger.info("Loaded model bundle from %s (v%s)", model_file.name,
                        _CACHED_BUNDLE.get("version", "unknown") if isinstance(_CACHED_BUNDLE, dict) else "legacy")
    except Exception as exc:
        logger.warning("Failed to load model from %s: %s", model_file, exc)
        return None, None

    return _CACHED_BUNDLE, _CACHED_PATH


# Backwards compat alias
def load_collaborator_model() -> Tuple[Optional[Any], Optional[Path]]:
    bundle, path = load_model_bundle()
    if bundle is None:
        return None, None
    # New bundle format: extract classifier
    if isinstance(bundle, dict) and "classifier" in bundle:
        return bundle["classifier"], path
    # Legacy format: bundle is the model itself
    return bundle, path


def _get_classifier_and_regressor(bundle):
    """Extract classifier and regressor from a model bundle."""
    if bundle is None:
        return None, None
    if isinstance(bundle, dict):
        return bundle.get("classifier"), bundle.get("regressor")
    # Legacy: bundle is the classifier directly
    if hasattr(bundle, "predict"):
        return bundle, None
    return None, None


# ---------------------------------------------------------------------------
# Risk score computation
# ---------------------------------------------------------------------------
def compute_risk_score(
    label: str,
    probabilities: Dict[str, float],
    gas_raw: float,
    temp_c: float,
    humidity: float,
    trend: Optional[Dict[str, Any]] = None,
) -> Tuple[float, str, str]:
    """Compute a composite risk score (0-100), risk level, and emoji indicator.

    The risk score combines:
    - Spoilage probability (primary signal, 60% weight)
    - Gas reading severity (20% weight)
    - Environmental stress (10% weight)
    - Trend acceleration (10% weight)
    """
    # Base risk from spoilage probability
    p_spoiled = probabilities.get("SPOILED", probabilities.get("spoiled", 0.0))
    p_ripening = probabilities.get("RIPENING", probabilities.get("ripe", probabilities.get("ripening", 0.0)))
    prob_risk = p_spoiled * 100.0 + p_ripening * 30.0  # Ripening adds moderate risk

    # Gas severity (normalized against known thresholds from training data)
    gas_risk = min(100, max(0, (gas_raw - 300) / 22.0))  # ~300=safe, ~2500=max risk

    # Environmental stress
    temp_stress = max(0, (temp_c - 25.0) * 4.0)  # Higher temp = faster spoilage
    humid_stress = max(0, (humidity - 70.0) * 2.0)  # High humidity accelerates
    env_risk = min(100, temp_stress + humid_stress)

    # Trend acceleration component
    trend_risk = 0.0
    if trend:
        if trend.get("deterioration_speed") == "ACCELERATING":
            trend_risk = 40.0
        elif trend.get("deterioration_speed") == "MODERATE":
            trend_risk = 20.0
        elif trend.get("deterioration_speed") == "SLOW":
            trend_risk = 10.0
        elif trend.get("deterioration_speed") == "IMPROVING":
            trend_risk = -10.0
        # Rising gas trend
        rate = trend.get("gas_rate", 0)
        if rate > 10:
            trend_risk += 15.0
        elif rate > 5:
            trend_risk += 8.0
    trend_risk = max(0, min(100, trend_risk))

    # Weighted combination
    risk_score = (prob_risk * 0.60 + gas_risk * 0.20 + env_risk * 0.10 + trend_risk * 0.10)
    risk_score = round(float(np.clip(risk_score, 0, 100)), 1)

    # Map to level
    for lo, hi, level, emoji in RISK_LEVELS:
        if lo <= risk_score < hi:
            return risk_score, level, emoji

    return risk_score, "CRITICAL", "🔴"


# ---------------------------------------------------------------------------
# Alerts / Early-warning system
# ---------------------------------------------------------------------------
def generate_alerts(
    label: str,
    risk_score: float,
    risk_level: str,
    shelf_life_days: Optional[float],
    trend: Optional[Dict[str, Any]],
    gas_raw: float,
    temp_c: float,
    transit: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    """Generate context-aware alerts based on the current state."""
    alerts = []

    if risk_level == "CRITICAL":
        alerts.append({"level": "CRITICAL", "icon": "🔴",
                        "message": "SPOILAGE DETECTED — banana has likely spoiled. Remove from storage."})
    elif risk_level == "HIGH":
        alerts.append({"level": "WARNING", "icon": "🟠",
                        "message": "HIGH SPOILAGE RISK — banana deteriorating rapidly. Consume soon."})

    if shelf_life_days is not None:
        if shelf_life_days < 0.5:
            alerts.append({"level": "URGENT", "icon": "⏰",
                            "message": "Less than 12 hours of shelf life remaining."})
        elif shelf_life_days < 1.0:
            alerts.append({"level": "WARNING", "icon": "⏳",
                            "message": f"Approximately {shelf_life_days * 24:.0f} hours of shelf life remaining."})
        elif shelf_life_days < 2.0:
            alerts.append({"level": "CAUTION", "icon": "📢",
                            "message": f"Estimated {shelf_life_days:.1f} days until spoilage."})

    if trend:
        if trend.get("deterioration_speed") == "ACCELERATING":
            alerts.append({"level": "WARNING", "icon": "📈",
                            "message": "Gas readings accelerating — spoilage may arrive sooner than estimated."})
        if trend.get("gas_trend") == "RISING_FAST":
            alerts.append({"level": "CAUTION", "icon": "⚡",
                            "message": f"Gas readings rising rapidly ({trend.get('gas_rate', 0):+.1f} ADC/reading)."})

    if temp_c > 30:
        alerts.append({"level": "INFO", "icon": "🌡️",
                        "message": f"High temperature ({temp_c:.1f}°C) accelerates ripening. Consider cooler storage."})

    if gas_raw > 2000:
        alerts.append({"level": "WARNING", "icon": "💨",
                        "message": f"Very high gas reading ({gas_raw} ADC) — strong VOC emission detected."})

    # MPU-6050 Transit Telemetry Alerts
    if transit:
        shock = transit.get("live_shock_g", 1.0)
        if shock >= 2.5:
            alerts.append({"level": "CRITICAL", "icon": "💥",
                            "message": f"SEVERE IMPACT DETECTED ({shock:.1f}g) — crate dropped! Bruising cuts shelf life."})
        elif shock >= 1.6:
            alerts.append({"level": "CAUTION", "icon": "⚠️",
                            "message": f"Transit bump recorded ({shock:.1f}g)."})

        tilt = transit.get("tilt_status", "")
        if "INVERTED" in tilt or "CRITICAL TILT" in tilt:
            alerts.append({"level": "WARNING", "icon": "📦",
                            "message": f"CRATE TILT HAZARD: {tilt} — risk of crushing fruit under crate weight."})

        if transit.get("road_condition") == "ROUGH_ROAD":
            alerts.append({"level": "CAUTION", "icon": "🚛",
                            "message": "Rough road transit active — ongoing vibration stress on fruit."})

        if transit.get("severe_drops", 0) > 0 and transit.get("bruise_penalty_percent", 0) > 0:
            alerts.append({"level": "WARNING", "icon": "📉",
                            "message": f"Shelf life reduced by {transit.get('bruise_penalty_percent', 0):.0f}% due to handling drops."})

    return alerts


# ---------------------------------------------------------------------------
# Heuristic fallback (no trained model)
# ---------------------------------------------------------------------------
def compute_baseline_prediction(gas_raw: float, temp_c: float, humidity: float, sample: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Domain-heuristic baseline for banana spoilage based on gas accumulation.

    Research-calibrated thresholds matching ESP32 MQ-3 in ambient air:
    - Ambient / fresh air: MQ-3 ADC < 1500 (label: "fresh")
    - Ripening banana VOC emission: MQ-3 ADC 1500-2300 (label: "ripe")
    - Rotten / spoiled banana alcohol surge: MQ-3 ADC > 2300 (label: "spoiled")
    """
    if gas_raw < 1500:
        label = "fresh"
        base_conf = 0.90 - (gas_raw / 1500.0) * 0.15
        risk_score = (gas_raw / 1500.0) * 28.0
        shelf_life = max(1.0, 5.0 - (gas_raw / 1500.0) * 2.5)
        probs = {"FRESH": round(base_conf, 4), "RIPENING": round(1.0 - base_conf, 4), "SPOILED": 0.0}
    elif gas_raw < 2300:
        label = "ripe"
        progress = (gas_raw - 1500.0) / (2300.0 - 1500.0)
        base_conf = 0.82 + progress * 0.10
        risk_score = 35.0 + progress * 35.0
        shelf_life = max(0.2, 2.5 - progress * 2.0)
        probs = {"FRESH": round(max(0.0, 0.2 - progress * 0.2), 4), "RIPENING": round(base_conf, 4), "SPOILED": round(max(0.0, 1.0 - base_conf), 4)}
    else:
        label = "spoiled"
        overflow = min(1.0, (gas_raw - 2300.0) / 1000.0)
        base_conf = 0.90 + overflow * 0.08
        risk_score = min(100.0, 75.0 + overflow * 25.0)
        shelf_life = max(0.0, 0.5 - overflow * 0.5)
        probs = {"FRESH": 0.0, "RIPENING": round(max(0.0, 0.1 - overflow * 0.1), 4), "SPOILED": round(base_conf, 4)}

    # Environmental adjustments
    temp_penalty = max(0.0, (temp_c - 25.0) * 0.8)
    humidity_penalty = max(0.0, (humidity - 70.0) * 0.3)
    adjusted_risk = min(100.0, max(0.0, risk_score + temp_penalty + humidity_penalty))

    if temp_c > 25:
        shelf_life *= max(0.3, 1.0 - (temp_c - 25) * 0.04)

    # Risk level & emoji
    for lo, hi, lvl, emj in RISK_LEVELS:
        if lo <= adjusted_risk < hi:
            risk_level, risk_emoji = lvl, emj
            break
    else:
        risk_level, risk_emoji = "CRITICAL", "🔴"

    trend = _trend_tracker.get_trend()
    
    # Process transit telemetry
    if sample:
        _transit_tracker.add(sample)
    transit = _transit_tracker.get_transit_state()

    # Biological freshness vs Handling integrity dual scores
    bio_freshness_score = round(max(0.0, min(100.0, 100.0 - adjusted_risk)), 1)
    handling_integrity_score = transit.get("handling_integrity_score", 100.0)
    composite_transit_score = round(0.60 * bio_freshness_score + 0.40 * handling_integrity_score, 1)

    # Shock & bruise degradation on shelf life
    bruise_penalty = transit.get("bruise_penalty_percent", 0.0) / 100.0
    shelf_life_baseline = round(shelf_life, 2)
    shelf_life_transit = round(max(0.05, shelf_life * (1.0 - bruise_penalty)), 2)

    alerts = generate_alerts(label, adjusted_risk, risk_level, shelf_life_transit, trend, gas_raw, temp_c, transit)

    return {
        "label": label,
        "confidence": float(np.clip(base_conf, 0.50, 0.99)),
        "probabilities": probs,
        "risk_score": round(float(adjusted_risk), 1),
        "risk_level": risk_level,
        "risk_emoji": risk_emoji,
        "shelf_life_days": shelf_life_transit,
        "shelf_life_days_baseline": shelf_life_baseline,
        "shelf_life_hours": round(shelf_life_transit * 24, 1),
        "bio_freshness_score": bio_freshness_score,
        "handling_integrity_score": handling_integrity_score,
        "composite_transit_score": composite_transit_score,
        "transit": transit,
        "trend": trend,
        "alerts": alerts,
        "model_source": "baseline_heuristic",
        "model_info": "Domain biochemical baseline heuristic (ambient ESP32 MQ-3 calibration)",
        "features": {"gas_raw": gas_raw, "temperature_c": temp_c, "humidity_percent": humidity},
    }


# ---------------------------------------------------------------------------
# Main prediction functions
# ---------------------------------------------------------------------------
def predict_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Predict spoilage state for a single sensor sample dictionary.

    Returns a rich result dict containing:
    - label: FRESH / RIPENING / SPOILED
    - confidence: 0.0-1.0
    - probabilities: {FRESH: ..., RIPENING: ..., SPOILED: ...}
    - risk_score: 0-100
    - risk_level: LOW / MEDIUM / HIGH / CRITICAL
    - shelf_life_days: estimated remaining shelf life
    - shelf_life_hours: same in hours
    - trend: sliding window trend analysis
    - alerts: list of contextual alerts
    """
    gas = float(sample.get("gas_raw", 0))
    temp = float(sample.get("temperature_c", 25.0))
    hum = float(sample.get("humidity_percent", 50.0))

    # Track trend and transit telemetry
    _trend_tracker.add(sample)
    trend = _trend_tracker.get_trend()
    _transit_tracker.add(sample)
    transit = _transit_tracker.get_transit_state()

    # Try loading model bundle
    bundle, path = load_model_bundle()
    classifier, regressor = _get_classifier_and_regressor(bundle)

    if classifier is not None:
        try:
            features = STANDARD_FEATURES
            if isinstance(bundle, dict) and "features" in bundle:
                features = bundle["features"]

            row = pd.DataFrame([{f: sample.get(f, 0) for f in features}])[features].apply(
                pd.to_numeric, errors="coerce"
            )

            if row.isna().any().any():
                raise ValueError("Invalid numeric values in sensor readings")

            # Classification
            label = str(classifier.predict(row)[0])
            probabilities: Dict[str, float] = {}
            confidence: Optional[float] = None

            if hasattr(classifier, "predict_proba"):
                probs_arr = classifier.predict_proba(row)[0]
                classes = [str(c) for c in getattr(classifier, "classes_", LABELS)]
                probabilities = {c: round(float(p), 4) for c, p in zip(classes, probs_arr)}
                confidence = float(probs_arr.max())

            # Shelf-life estimation
            shelf_life_days: Optional[float] = None
            if regressor is not None:
                try:
                    shelf_pred = regressor.predict(row)[0]
                    shelf_life_days = round(max(0.0, float(shelf_pred)), 2)
                except Exception as e:
                    logger.warning("Shelf-life prediction failed: %s", e)

            # If no regressor, estimate from probabilities
            if shelf_life_days is None and probabilities:
                p_fresh = probabilities.get("FRESH", 0)
                p_ripe = probabilities.get("RIPENING", 0)
                p_spoil = probabilities.get("SPOILED", 0)
                shelf_life_days = round(p_fresh * 5.0 + p_ripe * 2.0 + p_spoil * 0.2, 2)

            # Risk scoring
            risk_score, risk_level, risk_emoji = compute_risk_score(
                label, probabilities, gas, temp, hum, trend
            )

            # Dual scores & bruise adjustment
            bio_freshness_score = round(max(0.0, min(100.0, 100.0 - risk_score)), 1)
            handling_integrity_score = transit.get("handling_integrity_score", 100.0)
            composite_transit_score = round(0.60 * bio_freshness_score + 0.40 * handling_integrity_score, 1)

            shelf_life_days_baseline = shelf_life_days
            bruise_penalty = transit.get("bruise_penalty_percent", 0.0) / 100.0
            shelf_life_days_transit = (
                round(max(0.05, shelf_life_days * (1.0 - bruise_penalty)), 2)
                if shelf_life_days is not None
                else None
            )

            # Alerts with transit integration
            alerts = generate_alerts(label, risk_score, risk_level, shelf_life_days_transit, trend, gas, temp, transit)

            return {
                "label": label,
                "confidence": confidence,
                "probabilities": probabilities,
                "risk_score": risk_score,
                "risk_level": risk_level,
                "risk_emoji": risk_emoji,
                "shelf_life_days": shelf_life_days_transit if shelf_life_days_transit is not None else shelf_life_days,
                "shelf_life_days_baseline": shelf_life_days_baseline,
                "shelf_life_hours": round(shelf_life_days_transit * 24, 1) if shelf_life_days_transit is not None else None,
                "bio_freshness_score": bio_freshness_score,
                "handling_integrity_score": handling_integrity_score,
                "composite_transit_score": composite_transit_score,
                "transit": transit,
                "trend": trend,
                "alerts": alerts,
                "model_source": "trained_model",
                "model_file": path.name if path else "model.joblib",
                "model_version": bundle.get("version", "unknown") if isinstance(bundle, dict) else "legacy",
                "features": {"gas_raw": gas, "temperature_c": temp, "humidity_percent": hum},
            }
        except Exception as exc:
            logger.warning("Model inference failed: %s; falling back to heuristic", exc)

    # Fallback to heuristic
    return compute_baseline_prediction(gas, temp, hum, sample=sample)


def predict_dataframe(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Predict spoilage state from the latest row of a DataFrame."""
    if df.empty:
        return None

    latest_row = df.iloc[-1]
    sample = {
        "gas_raw": float(latest_row["gas_raw"]),
        "temperature_c": float(latest_row["temperature_c"]),
        "humidity_percent": float(latest_row["humidity_percent"]),
    }
    for k in ["accel_x", "accel_y", "accel_z", "shock_g", "pitch_deg", "roll_deg", "timestamp"]:
        if k in latest_row and pd.notna(latest_row[k]):
            sample[k] = latest_row[k]

    return predict_sample(sample)
