"""SporeWatch — banana freshness monitor (stage 1, DEMO MODE).

Run with:  python -m streamlit run app.py

This stage is UI only: it never touches hardware, a real dataset or a trained
model. All measurements shown are synthetic and labelled as such.
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

st.set_page_config(page_title="SporeWatch — Banana Freshness Monitor", page_icon="🍌", layout="wide", initial_sidebar_state="collapsed")


@st.cache_data(show_spinner=False)
def build_demo_dataset(seed: int = T.DEMO_SEED, rows: int = T.DEMO_ROWS) -> pd.DataFrame:
    """Small, deterministic, in-memory synthetic dataset. Never written to disk."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range(T.DEMO_START, periods=rows, freq=f"{T.DEMO_INTERVAL_MIN}min")
    t = np.arange(rows)
    mq = 1620 + 3.2 * t + 38 * np.sin(t / 3.1) + rng.normal(0, 11, rows)       # raw ADC counts (0-4095)
    temp = 25.2 + 0.9 * np.sin(t / 7.5) + rng.normal(0, 0.12, rows)            # deg C
    hum = 64.0 + 2.6 * np.cos(t / 6.0) + rng.normal(0, 0.35, rows)             # % RH
    return pd.DataFrame(
        {
            "timestamp": ts,
            "specimen_id": T.SPECIMEN_ID,
            "gas_raw": np.rint(mq).astype(int),
            "temperature_c": np.round(temp, 1),
            "humidity_percent": np.round(hum, 1),
            "source": T.DATA_SOURCE,
        }
    )


def _col():
    from pymongo import MongoClient

    return MongoClient(os.environ.get("MONGO_URI", "mongodb://localhost:27017"), serverSelectionTimeoutMS=2000).freshguard.readings


@st.cache_data(ttl=3, show_spinner=False)
def list_specimens() -> list[str]:
    """Specimen ids in MongoDB, most recently updated first (so the live one is the default)."""
    try:
        return [d["_id"] for d in _col().aggregate([{"$group": {"_id": "$specimen_id", "t": {"$max": "$timestamp"}}}, {"$sort": {"t": -1}}])]
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


def _activity_log(df: pd.DataFrame, started: datetime, live: bool) -> list[tuple[str, str, str, str]]:
    now = f"{started:%Y-%m-%d %H:%M:%S}"
    kind = "Live ESP32" if live else "Synthetic"
    return [
        (now, "INFO", "cyan", "Dashboard started." if live else "Dashboard initialized in DEMO MODE."),
        (now, "INFO", "cyan", f"{kind} dataset loaded: {len(df)} rows from {df.attrs.get('origin', 'unknown')}."),
        (now, "INFO" if live else "PENDING", "cyan" if live else "amber", "ESP32 readings are streaming via MongoDB." if live else "Hardware integration pending. No sensor connection has been attempted."),
        (now, "UNAVAILABLE", "crimson", "ML model unavailable: not trained. No spoilage prediction is produced."),
    ]


@st.fragment(run_every=3)
def dashboard(specimen: str | None) -> None:
    df = load_readings(specimen)
    latest = df.iloc[-1]
    live = latest.get("source") == SR.SOURCE_LABEL
    sid = str(latest.specimen_id)
    last_ts = f"{latest.timestamp:%Y-%m-%d %H:%M:%S}" + ("" if live else " (synthetic)")
    pred = predictor.predict(df) if live else None

    ui.render_hero(
        ui.SpecimenInfo(
            specimen_id=sid,
            experiment_id="EXP-LIVE" if live else T.EXPERIMENT_ID,
            monitoring_status="LIVE · ESP32" if live else T.MONITORING_STATUS,
            observation_count=len(df),
            last_observation=last_ts,
            data_source="ESP32 MQ-3 + DHT22" if live else T.DATA_SOURCE,
        )
    )
    ui.render_readouts(latest, total=len(df))

    left, right = st.columns([1.55, 1])
    with left:
        ui.render_assessment(sid, pred is not None, f"{st.session_state.started_at:%Y-%m-%d %H:%M:%S}", pred, live)
    with right:
        ui.render_system_status(
            [("HARDWARE", "ESP32 CONNECTED", "lime"), ("DATA SOURCE", "LIVE SENSOR", "lime"), ("DATASET", "STREAMING", "lime"),
             ("ML MODEL", "ACTIVE" if pred else "NOT TRAINED", "lime" if pred else "crimson"), ("EXPERIMENTAL VALIDATION", "PENDING", "amber")]
            if live else T.SYSTEM_STATE
        )

    chart_col, table_col = st.columns([1, 1.15])
    with chart_col:
        ui.render_signal_trace(df)
    with table_col:
        ui.render_observations(df)

    ui.render_activity(_activity_log(df, st.session_state.started_at, live))


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
