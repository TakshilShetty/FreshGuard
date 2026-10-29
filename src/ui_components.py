"""SporeWatch — reusable UI components (PeelLab banana redesign, presentation only).

Everything here is presentation only. No sensor, dataset or model logic lives
in this module; callers pass in plain values.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Sequence

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui_theme as T
from src.ui_banana import background_layer, banana_icon


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _html(markup: str) -> str:
    """Strip indentation/blank lines so Markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in markup.splitlines() if line.strip())


def _render(markup: str) -> None:
    st.markdown(_html(markup), unsafe_allow_html=True)


def _e(value: object) -> str:
    return html.escape(str(value))


def tag(text: str, tone: str = "", dot: bool = False, pulse: bool = False) -> str:
    """Small uppercase status chip. tone: lime | cyan | amber | crimson | ''."""
    d = f'<span class="dot{" pulse" if pulse else ""}"></span>' if dot else ""
    return f'<span class="sw-tag {tone}">{d}{_e(text)}</span>'


def panel(title: str, index: str, body: str, right_tag: str = "", anchor: str = "") -> str:
    right = f'<span class="tag">{right_tag}</span>' if right_tag else ""
    aid = f' id="{anchor}"' if anchor else ""
    return (
        f'<div class="sw-panel"{aid}><div class="sw-title"><span class="idx">{_e(index)}</span>'
        f"{_e(title)}{right}</div>{body}</div>"
    )


def inject_css() -> None:
    """Load the single theme stylesheet."""
    css = T.CSS_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    _render(background_layer())


# --------------------------------------------------------------------------- #
# header
# --------------------------------------------------------------------------- #
def _utc_offset(now: datetime) -> str:
    off = now.utcoffset()
    mins = int(off.total_seconds() // 60) if off else 0
    sign = "+" if mins >= 0 else "-"
    return f"{sign}{abs(mins) // 60:02d}:{abs(mins) % 60:02d}"


def _clock_markup() -> str:
    now = datetime.now().astimezone()
    return (
        f'<div class="sw-hval">{now:%Y-%m-%d  %H:%M:%S}'
        f' <span style="color:var(--muted)">UTC{_utc_offset(now)}</span></div>'
    )


def render_header() -> None:
    """Banana wordmark + in-page navigation + honest status chips."""
    clock = _clock_markup()
    _render(
        f"""
        <div class="sw-header">
          <div class="sw-brand">{banana_icon()}<div>
            <div class="sw-wordmark">Spore<span class="sw-mark">Watch</span></div>
            <div class="sw-descriptor">{_e(T.DESCRIPTOR)}</div></div></div>
          <nav class="sw-nav" aria-label="Page sections">
            <a href="#sw-specimen">Specimen</a><a href="#sw-assess">Assessment</a>
            <a href="#sw-obs">Observations</a><a href="#sw-activity">Activity</a></nav>
          <div class="sw-hstatus">
            <div class="sw-hcell"><span class="sw-label">Local time</span>{clock}</div>
            <div class="sw-hcell"><span class="sw-label">Data source</span>{tag("Simulated", "cyan")}</div>
            <div class="sw-demo-badge">DEMO MODE</div>
          </div>
        </div>
        """
    )


# --------------------------------------------------------------------------- #
# hero
# --------------------------------------------------------------------------- #
_SPECIMEN_SVG = banana_icon()


@dataclass(frozen=True)
class SpecimenInfo:
    specimen_id: str
    experiment_id: str
    monitoring_status: str
    observation_count: int
    last_observation: Optional[str]
    data_source: str


def render_hero(info: SpecimenInfo) -> None:
    last = _e(info.last_observation) if info.last_observation else "UNAVAILABLE"
    prefix, _, num = info.specimen_id.rpartition("-")
    _render(
        f"""
        <div class="sw-panel" id="sw-specimen">
          <div class="sw-hero">
            <div class="sw-hero-left">
              <div>
                <div style="display:flex;gap:.5rem;flex-wrap:wrap;align-items:center">
                  <span class="sw-label">Selected specimen</span>
                  {tag(info.monitoring_status, "amber", dot=True, pulse=True)}
                </div>
                <div class="sw-h1">Bananas don&rsquo;t spoil in <span class="sw-mark">silence.</span></div>
                <div class="sw-hero-id">{_e(prefix)}-<span>{_e(num)}</span></div>
                <div class="sw-hero-exp">EXPERIMENT // {_e(info.experiment_id)}</div>
                <div class="sw-explain" style="margin-top:.8rem;max-width:32rem">Specimen under <b>simulated</b> observation. No physical sensor is attached and no real measurements are being recorded.</div>
              </div>
              <div class="sw-meta">
                <div><span class="sw-label">Monitoring status</span><div class="v">{_e(info.monitoring_status)}</div></div>
                <div><span class="sw-label">Data source</span><div class="v">{_e(info.data_source)}</div></div>
                <div><span class="sw-label">Observation count</span><div class="v">{info.observation_count:,} <span class="sw-syn">SYNTHETIC</span></div></div>
                <div><span class="sw-label">Last observation</span><div class="v">{last}</div></div>
              </div>
            </div>
            <div class="sw-hero-art">{_SPECIMEN_SVG}
              <div class="sw-sim-note">{tag("Illustration only", "")}</div>
            </div>
          </div>
        </div>
        """
    )


# --------------------------------------------------------------------------- #
# sensor readouts
# --------------------------------------------------------------------------- #
def readout_card(
    label: str,
    value: Optional[float | int],
    unit: str = "",
    note: str = "",
    decimals: int = 0,
    fill: Optional[float] = None,
    tone: str = "",
    demo: bool = True,
) -> str:
    """One large monospace readout. ``value=None`` renders a polished unavailable state."""
    if value is None:
        val = '<div class="val na">— — —</div>'
        state = tag("Unavailable", "amber")
    else:
        u = f"<small>{_e(unit)}</small>" if unit else ""
        val = f'<div class="val">{value:,.{decimals}f}{u}</div>'
        state = tag("Demo data", "amber", dot=True) if demo else tag("Measured", "lime", dot=True)
    bar = f'<div class="bar"><i style="width:{max(0, min(100, fill)):.0f}%"></i></div>' if fill is not None and value is not None else '<div class="bar"></div>'
    return (
        f'<div class="sw-panel sw-readout {tone}"><div style="display:flex;justify-content:space-between;gap:.5rem">'
        f'<span class="sw-label">{_e(label)}</span></div>{val}{bar}'
        f'<div class="foot"><span class="note">{_e(note)}</span>{state}</div></div>'
    )


def render_readouts(latest: Optional[pd.Series], total: int) -> None:
    cols = st.columns(4)
    mq = temp = hum = None
    if latest is not None:
        mq, temp, hum = latest["gas_raw"], latest["temperature_c"], latest["humidity_percent"]
    cards = [
        readout_card("MQ-135 · raw sensor reading", mq, "ADC", "Raw counts · not ppm", 0, (mq / 4095 * 100) if mq is not None else None, "lime"),
        readout_card("Temperature", temp, "°C", "Simulated placeholder", 1, (temp / 50 * 100) if temp is not None else None, "coral"),
        readout_card("Relative humidity", hum, "% RH", "Simulated placeholder", 1, hum, "sky"),
        readout_card("Total observations", total, "", "Synthetic rows only", 0, None, "mustard"),
    ]
    for col, card in zip(cols, cards):
        with col:
            _render(card)


# --------------------------------------------------------------------------- #
# assessment
# --------------------------------------------------------------------------- #
def render_assessment(specimen_id: str, model_trained: bool, timestamp: str) -> None:
    """Assessment panel. Shows the unavailable state until a validated model exists.

    A later stage can replace the ``sw-slot`` block with a real assessment; the
    panel intentionally never renders percentages or confidence on its own.
    """
    body = f"""
    <div class="sw-assess">
      <div class="big">{"MODEL AVAILABLE" if model_trained else "MODEL NOT TRAINED"}</div>
      <div class="sub">NO VALIDATED SPOILAGE PREDICTION AVAILABLE</div>
    </div>
    <div class="sw-assess-grid">
      <div><span class="sw-label">Assessment status</span><div class="v">{tag("Not available", "amber")}</div></div>
      <div><span class="sw-label">Model availability</span><div class="v">{tag("Available" if model_trained else "Unavailable", "lime" if model_trained else "crimson")}</div></div>
      <div><span class="sw-label">Specimen</span><div class="v">{_e(specimen_id)}</div></div>
      <div><span class="sw-label">Timestamp</span><div class="v">{_e(timestamp)}</div></div>
      <div><span class="sw-label">Prediction</span><div class="v" style="color:#6F6A60">— — —</div></div>
      <div><span class="sw-label">Confidence</span><div class="v" style="color:#6F6A60">— — —</div></div>
    </div>
    <div class="sw-explain">
      <b>Why there is no assessment.</b> No model has been trained and no real dataset has been collected,
      so the system has nothing validated to base a freshness call on. The MQ-135 is a general
      air-quality gas sensor, not a banana-specific spoilage detector &mdash; a changing reading alone
      cannot show that fruit is spoiled. Values on this screen are <b>simulated</b> and are never
      interpreted as fresh or spoiled.
    </div>
    <div class="sw-slot"><span class="sw-label">Assessment output slot</span>{tag("Reserved · awaiting validated model", "amber")}</div>
    """
    _render(panel("Freshness / spoilage assessment", "03", body, tag("No prediction", "amber"), "sw-assess"))


# --------------------------------------------------------------------------- #
# system status
# --------------------------------------------------------------------------- #
def render_system_status(states: Sequence[tuple[str, str, str]]) -> None:
    rows = "".join(
        f'<div class="sw-status-row"><span class="k">{_e(k)}</span>{tag(v, tone, dot=True)}</div>'
        for k, v, tone in states
    )
    _render(panel("System status", "04", rows))


# --------------------------------------------------------------------------- #
# observations, chart, activity
# --------------------------------------------------------------------------- #
def render_observations(df: pd.DataFrame, limit: int = 10) -> None:
    recent = df.sort_values("timestamp", ascending=False).head(limit)
    rows = "".join(
        f"<tr><td>{r.timestamp:%Y-%m-%d %H:%M}</td><td>{_e(r.specimen_id)}</td>"
        f'<td class="num">{r.gas_raw:,}</td><td class="num">{r.temperature_c:.1f} °C</td>'
        f'<td class="num">{r.humidity_percent:.1f} %</td><td class="src">{_e(r.source)}</td></tr>'
        for r in recent.itertuples()
    )
    table = (
        '<div class="sw-scroll"><table class="sw-table"><thead><tr><th>Timestamp</th><th>Specimen</th>'
        '<th class="num">MQ-135 raw</th><th class="num">Temp</th><th class="num">Humidity</th><th>Source</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        '<div class="sw-label" style="margin-top:.6rem">Synthetic demo dataset · fixed seed · generated in memory · not from the ESP32</div>'
    )
    _render(panel("Recent observations", "05", table, tag("Synthetic", "amber"), "sw-obs"))


def _rgba(hex_color: str, a: float = 0.14) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _trace(df: pd.DataFrame, column: str, name: str, color: str, unit: str) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=df["timestamp"], y=df[column], mode="lines", name=name,
            line=dict(color=color, width=3, shape="spline", smoothing=0.5),
            fill="tozeroy", fillcolor=_rgba(color),
            hovertemplate=f"%{{x|%H:%M}}<br>%{{y:,.1f}} {unit}<extra>SYNTHETIC</extra>",
        )
    )
    lo, hi = float(df[column].min()), float(df[column].max())
    pad = max((hi - lo) * 0.35, 1e-6)
    fig.update_layout(
        height=250, margin=dict(l=48, r=14, t=10, b=34), showlegend=False,
        paper_bgcolor=T.PANEL, plot_bgcolor=T.PANEL, font=dict(family="DM Sans, Segoe UI, sans-serif", color=T.TEXT, size=11),
        xaxis=dict(gridcolor="#E4DCC8", linecolor=T.TEXT, tickformat="%H:%M", title=None),
        yaxis=dict(gridcolor="#E4DCC8", linecolor=T.TEXT, range=[lo - pad, hi + pad], title=dict(text=unit, font=dict(size=10))),
        hoverlabel=dict(bgcolor=T.MUSTARD, bordercolor=T.TEXT, font=dict(family="DM Sans, Segoe UI, sans-serif", color=T.TEXT)),
    )
    return fig


def render_signal_trace(df: pd.DataFrame) -> None:
    with st.container(border=True):
        _render(
            '<div class="sw-title" style="margin-bottom:.4rem"><span class="idx">06</span>Signal trace'
            f'<span class="tag">{tag("Synthetic · not real measurements", "amber")}</span></div>'
        )
        t1, t2, t3 = st.tabs(["MQ-135 RAW", "TEMPERATURE", "HUMIDITY"])
        cfg = {"displayModeBar": False}
        with t1:
            st.plotly_chart(_trace(df, "gas_raw", "MQ-135 raw", T.TEAL, "ADC counts"), width="stretch", config=cfg)
        with t2:
            st.plotly_chart(_trace(df, "temperature_c", "Temperature", T.CORAL, "°C"), width="stretch", config=cfg)
        with t3:
            st.plotly_chart(_trace(df, "humidity_percent", "Humidity", T.SKY, "% RH"), width="stretch", config=cfg)


def render_activity(events: Sequence[tuple[str, str, str, str]]) -> None:
    """events: (timestamp, level label, tone, message)."""
    rows = "".join(
        f'<div class="row"><span class="ts">{_e(ts)}</span>{tag(lvl, tone)}<span class="msg">{_e(msg)}</span></div>'
        for ts, lvl, tone, msg in events
    )
    _render(panel("System activity", "07", f'<div class="sw-log">{rows}</div>', anchor="sw-activity"))


# --------------------------------------------------------------------------- #
# footer
# --------------------------------------------------------------------------- #
def footer_note() -> None:
    _render(
        '<div class="sw-label" style="text-align:center;margin-top:1rem;opacity:.7">'
        "SporeWatch · stage 1 · demo interface · no hardware · no trained model · simulated data only</div>"
    )
