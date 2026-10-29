"""SporeWatch — banana freshness monitor (stage 1, DEMO MODE).

Run with:  python -m streamlit run app.py

This stage is UI only: it never touches hardware, a real dataset or a trained
model. All measurements shown are synthetic and labelled as such.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from src import ui_theme as T
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


def _activity_log(df: pd.DataFrame, started: datetime) -> list[tuple[str, str, str, str]]:
    now = f"{started:%Y-%m-%d %H:%M:%S}"
    return [
        (now, "INFO", "cyan", "Dashboard initialized in DEMO MODE."),
        (now, "INFO", "cyan", f"Synthetic demo dataset loaded: {len(df)} rows, seed {T.DEMO_SEED}, held in memory only."),
        (now, "PENDING", "amber", "Hardware integration pending. No sensor connection has been attempted."),
        (now, "PENDING", "amber", "No real dataset has been collected."),
        (now, "UNAVAILABLE", "crimson", "ML model unavailable: not trained. No spoilage prediction is produced."),
    ]


def main() -> None:
    ui.inject_css()

    if "started_at" not in st.session_state:
        st.session_state.started_at = datetime.now().astimezone()

    df = build_demo_dataset()
    latest = df.iloc[-1]
    last_ts = f"{latest.timestamp:%Y-%m-%d %H:%M:%S} (synthetic)"

    ui.render_header()

    ui.render_hero(
        ui.SpecimenInfo(
            specimen_id=T.SPECIMEN_ID,
            experiment_id=T.EXPERIMENT_ID,
            monitoring_status=T.MONITORING_STATUS,
            observation_count=len(df),
            last_observation=last_ts,
            data_source=T.DATA_SOURCE,
        )
    )

    ui.render_readouts(latest, total=len(df))

    left, right = st.columns([1.55, 1])
    with left:
        ui.render_assessment(T.SPECIMEN_ID, T.MODEL_TRAINED, f"{st.session_state.started_at:%Y-%m-%d %H:%M:%S}")
    with right:
        ui.render_system_status(T.SYSTEM_STATE)

    chart_col, table_col = st.columns([1, 1.15])
    with chart_col:
        ui.render_signal_trace(df)
    with table_col:
        ui.render_observations(df)

    ui.render_activity(_activity_log(df, st.session_state.started_at))
    ui.footer_note()


main()
