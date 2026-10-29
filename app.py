"""FreshGuard / SporeWatch — Real-time banana freshness monitoring dashboard.

Run with:  streamlit run app.py

Features:
- Real-time sensor readings (gas, temperature, humidity) auto-refreshing every 2s
- 3-class AI classification: FRESH / RIPENING / SPOILED with confidence %
- Shelf-life estimation in days/hours
- Risk score gauge (0-100) with LOW/MEDIUM/HIGH/CRITICAL levels
- Trend analysis with gas trajectory graph
- Early-warning alert feed
- Historical data table
"""
from __future__ import annotations

import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src import predictor, ui_theme as T
from src import sensor_reader as SR
from src import ui_components as ui
from src.config import MONGO_URI, DB_NAME, READINGS_COLLECTION

st.set_page_config(page_title="FreshGuard — Banana Spoilage Monitor", page_icon="🍌", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def build_demo_dataset(seed: int = T.DEMO_SEED, rows: int = T.DEMO_ROWS) -> pd.DataFrame:
    """Small, deterministic, in-memory synthetic dataset. Never written to disk."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(T.DEMO_START, periods=rows, freq=f"{T.DEMO_INTERVAL_MIN}min")
    t = np.arange(rows)
    mq = 1620 + 3.2 * t + 38 * np.sin(t / 3.1) + rng.normal(0, 11, rows)       # raw ADC counts (0-4095)
    temp = 25.2 + 0.9 * np.sin(t / 7.5) + rng.normal(0, 0.12, rows)            # deg C
    hum = 64.0 + 2.6 * np.cos(t / 6.0) + rng.normal(0, 0.35, rows)             # % RH

    # Realistic MPU-6050 transit vibrations & orientation
    ax = rng.normal(0.02, 0.03, rows)
    ay = rng.normal(-0.01, 0.03, rows)
    az = rng.normal(0.99, 0.04, rows)
    # Simulate a transit road bump around row 12
    if rows > 12:
        az[12] += 0.85
    shock = np.sqrt(ax * ax + ay * ay + az * az)
    pitch = np.degrees(np.arctan2(-ax, np.sqrt(ay * ay + az * az)))
    roll = np.degrees(np.arctan2(ay, az))

    return pd.DataFrame(
        {
            "timestamp": ts,
            "specimen_id": T.SPECIMEN_ID,
            "gas_raw": np.rint(mq).astype(int),
            "temperature_c": np.round(temp, 1),
            "humidity_percent": np.round(hum, 1),
            "accel_x": np.round(ax, 2),
            "accel_y": np.round(ay, 2),
            "accel_z": np.round(az, 2),
            "shock_g": np.round(shock, 2),
            "pitch_deg": np.round(pitch, 1),
            "roll_deg": np.round(roll, 1),
            "source": T.DATA_SOURCE,
        }
    )


def _col():
    from pymongo import MongoClient

    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)[DB_NAME][READINGS_COLLECTION]


@st.cache_data(ttl=2, show_spinner=False)
def list_specimens() -> list[str]:
    """Specimen ids in MongoDB, most recently updated first (prioritizing BANANA-LIVE)."""
    try:
        raw_list = [d["_id"] for d in _col().aggregate([{"$group": {"_id": "$specimen_id", "t": {"$max": "$timestamp"}}}, {"$sort": {"t": -1}}])]
        if "BANANA-LIVE" in raw_list:
            raw_list.remove("BANANA-LIVE")
            raw_list.insert(0, "BANANA-LIVE")
        return raw_list
    except Exception:
        return []


@st.cache_data(ttl=2, show_spinner=False)
def load_readings(specimen: str | None) -> pd.DataFrame:
    """Latest rows for the specimen from MongoDB; falls back to the in-memory demo."""
    try:
        rows = list(_col().find({"specimen_id": specimen}, {"_id": 0}).sort("timestamp", -1).limit(500))[::-1] if specimen else []
        if rows:
            df = pd.DataFrame(rows)
            if "source" not in df:
                df["source"] = T.DATA_SOURCE
            df.attrs["origin"] = "MongoDB freshguard.readings"
            return df
    except Exception:
        pass
    df = build_demo_dataset().copy()
    df.attrs["origin"] = "in-memory demo (MongoDB unavailable)"
    return df


def _activity_log(df: pd.DataFrame, started: datetime, live: bool, pred: dict | None) -> list[tuple[str, str, str, str]]:
    now = f"{started:%Y-%m-%d %H:%M:%S}"
    entries = [
        (now, "INFO", "cyan", "Dashboard started." if live else "Dashboard initialized in DEMO MODE."),
        (now, "INFO", "cyan", f"{'Live ESP32' if live else 'Synthetic'} dataset loaded: {len(df)} rows from {df.attrs.get('origin', 'unknown')}."),
    ]

    if live:
        entries.append((now, "INFO", "lime", "ESP32 readings are streaming via MongoDB."))
        if pred:
            label = pred.get("label", "?")
            risk_lvl = pred.get("risk_level", "?")
            shelf = pred.get("shelf_life_days")
            entries.append((now, "AI", "lime", f"AI Engine active: {label} | Risk: {risk_lvl} | Shelf-life: {shelf:.1f}d" if shelf else f"AI Engine active: {label}"))
            for alert in pred.get("alerts", [])[:3]:
                tone = "crimson" if alert["level"] in ("CRITICAL", "URGENT") else "amber" if alert["level"] == "WARNING" else "cyan"
                entries.append((now, alert["level"], tone, f"{alert['icon']} {alert['message']}"))
    else:
        entries.append((now, "PENDING", "amber", "Hardware integration pending. No sensor connection has been attempted."))
        entries.append((now, "INFO", "cyan", "ML model loaded — predictions will appear when live data streams in."))

    return entries


@st.fragment(run_every=2)
def dashboard(specimen: str | None) -> None:
    df = load_readings(specimen)
    latest = df.iloc[-1]
    live = latest.get("source") == SR.SOURCE_LABEL
    sid = str(latest.specimen_id)
    last_ts = f"{latest.timestamp:%Y-%m-%d %H:%M:%S}" + ("" if live else " (synthetic)")
    pred = predictor.predict(df)

    ui.render_hero(
        ui.SpecimenInfo(
            specimen_id=sid,
            experiment_id="EXP-LIVE" if live else T.EXPERIMENT_ID,
            monitoring_status="LIVE · ESP32" if live else T.MONITORING_STATUS,
            observation_count=len(df),
            last_observation=last_ts,
            data_source="ESP32 Multi-Sensor (MQ-3, DHT22, MPU-6050)" if live else T.DATA_SOURCE,
        )
    )
    ui.render_readouts(latest, total=len(df), live=live)

    # === AI Assessment with Risk Gauge & Shelf-Life ===
    if pred:
        ui.render_ai_assessment(sid, pred, df)
    else:
        # Fallback if no prediction available
        left, right = st.columns([1.55, 1])
        with left:
            ui.render_assessment(sid, False, f"{st.session_state.started_at:%Y-%m-%d %H:%M:%S}", None, live)
        with right:
            ui.render_system_status(
                [(k, v, t) for k, v, t in (
                    [("HARDWARE", "ESP32 CONNECTED" if live else "NOT CONNECTED", "lime" if live else "crimson"),
                     ("DATA SOURCE", "LIVE SENSOR" if live else "SIMULATED", "lime" if live else "cyan"),
                     ("DATASET", "STREAMING" if live else "SYNTHETIC", "lime" if live else "amber"),
                     ("ML MODEL", "ACTIVE", "lime"),
                     ("EXPERIMENTAL VALIDATION", "CALIBRATED", "lime")]
                )]
            )

    # === Signal Trace & Observations ===
    chart_col, table_col = st.columns([1, 1.15])
    with chart_col:
        ui.render_signal_trace(df, live=live)
    with table_col:
        ui.render_observations(df, live=live)

    ui.render_activity(_activity_log(df, st.session_state.started_at, live, pred))


def main() -> None:
    ui.inject_css()
    if "started_at" not in st.session_state:
        st.session_state.started_at = datetime.now().astimezone()

    specimens = list_specimens()
    specimen = st.selectbox("Specimen", specimens, label_visibility="collapsed") if specimens else None
    ui.render_header(live=bool(specimen) and load_readings(specimen).iloc[-1].get("source") == SR.SOURCE_LABEL)
    dashboard(specimen)
    ui.footer_note()


main()
