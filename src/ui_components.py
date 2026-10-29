"""FreshGuard / SporeWatch — reusable UI components with real-time AI-powered displays.

Includes:
- Risk gauge (0-100) with color gradient
- Shelf-life countdown
- Confidence probability bars (FRESH/RIPENING/SPOILED)
- AI assessment panel with predictions
- Trend analysis chart
- Alert feed
- Sensor readout cards
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
    # Extra styles for AI components
    st.markdown("""<style>
    .risk-gauge { text-align: center; padding: 1rem 0; }
    .risk-gauge .score { font-size: 3rem; font-weight: 800; font-family: 'DM Sans', sans-serif; }
    .risk-gauge .level { font-size: 1.1rem; font-weight: 600; text-transform: uppercase; margin-top: 0.3rem; }
    .risk-low { color: #12B3A4; }
    .risk-medium { color: #FFC531; }
    .risk-high { color: #FF8C00; }
    .risk-critical { color: #FF5B57; }
    .shelf-life { text-align: center; padding: 0.8rem 0; }
    .shelf-life .days { font-size: 2.5rem; font-weight: 800; font-family: 'DM Sans', sans-serif; color: #17140D; }
    .shelf-life .hours { font-size: 1.2rem; color: #6F6A60; margin-top: 0.2rem; }
    .shelf-life .label { font-size: 0.85rem; color: #6F6A60; text-transform: uppercase; letter-spacing: 0.05em; }
    .prob-bar { display: flex; align-items: center; gap: 0.5rem; margin: 0.35rem 0; font-size: 0.85rem; }
    .prob-bar .bar-label { width: 5rem; text-align: right; font-weight: 600; font-family: 'DM Sans', sans-serif; }
    .prob-bar .bar-track { flex: 1; height: 1.2rem; background: #F0EBE0; border-radius: 0.6rem; overflow: hidden; }
    .prob-bar .bar-fill { height: 100%; border-radius: 0.6rem; transition: width 0.6s ease; }
    .prob-bar .bar-pct { width: 3.5rem; font-weight: 700; font-family: 'DM Sans', monospace; }
    .alert-item { display: flex; gap: 0.5rem; padding: 0.5rem 0.7rem; margin: 0.3rem 0; border-radius: 0.5rem; font-size: 0.85rem; }
    .alert-critical { background: rgba(255,91,87,0.12); border-left: 3px solid #FF5B57; }
    .alert-warning { background: rgba(255,140,0,0.10); border-left: 3px solid #FF8C00; }
    .alert-caution { background: rgba(255,197,49,0.10); border-left: 3px solid #FFC531; }
    .alert-info { background: rgba(18,179,164,0.08); border-left: 3px solid #12B3A4; }
    .alert-urgent { background: rgba(255,91,87,0.15); border-left: 3px solid #FF5B57; }
    </style>""", unsafe_allow_html=True)
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


def render_header(live: bool = False) -> None:
    """Banana wordmark + in-page navigation + honest status chips."""
    clock = _clock_markup()
    _render(
        f"""
        <div class="sw-header">
          <div class="sw-brand">{banana_icon()}<div>
            <div class="sw-wordmark">Fresh<span class="sw-mark">Guard</span></div>
            <div class="sw-descriptor">{_e(T.DESCRIPTOR)}</div></div></div>
          <nav class="sw-nav" aria-label="Page sections">
            <a href="#sw-specimen">Specimen</a><a href="#sw-assess">Assessment</a>
            <a href="#sw-obs">Observations</a><a href="#sw-activity">Activity</a></nav>
          <div class="sw-hstatus">
            <div class="sw-hcell"><span class="sw-label">Local time</span>{clock}</div>
            <div class="sw-hcell"><span class="sw-label">Data source</span>{tag("ESP32 live", "lime") if live else tag("Simulated", "cyan")}</div>
            <div class="sw-demo-badge">{"LIVE" if live else "DEMO MODE"}</div>
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
                  {tag(info.monitoring_status, "amber" if "DEMO" in info.monitoring_status else "lime", dot=True, pulse=True)}
                </div>
                <div class="sw-h1">Bananas don&rsquo;t spoil in <span class="sw-mark">silence.</span></div>
                <div class="sw-hero-id">{_e(prefix)}-<span>{_e(num)}</span></div>
                <div class="sw-hero-exp">EXPERIMENT // {_e(info.experiment_id)}</div>
                <div class="sw-explain" style="margin-top:.8rem;max-width:32rem">{"Specimen under <b>live</b> observation: real ESP32 sensor readings with AI-powered spoilage analysis, refreshed every 2 seconds." if info.data_source != T.DATA_SOURCE else "Specimen under <b>simulated</b> observation. No physical sensor is attached and no real measurements are being recorded."}</div>
              </div>
              <div class="sw-meta">
                <div><span class="sw-label">Monitoring status</span><div class="v">{_e(info.monitoring_status)}</div></div>
                <div><span class="sw-label">Data source</span><div class="v">{_e(info.data_source)}</div></div>
                <div><span class="sw-label">Observation count</span><div class="v">{info.observation_count:,}</div></div>
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
        state = tag("Measured", "lime", dot=True) if not demo else tag("Demo data", "amber", dot=True)
    bar = f'<div class="bar"><i style="width:{max(0, min(100, fill)):.0f}%"></i></div>' if fill is not None and value is not None else '<div class="bar"></div>'
    return (
        f'<div class="sw-panel sw-readout {tone}"><div style="display:flex;justify-content:space-between;gap:.5rem">'
        f'<span class="sw-label">{_e(label)}</span></div>{val}{bar}'
        f'<div class="foot"><span class="note">{_e(note)}</span>{state}</div></div>'
    )


def render_readouts(latest: Optional[pd.Series], total: int, live: bool = False) -> None:
    cols = st.columns(4)
    mq = temp = hum = None
    if latest is not None:
        mq, temp, hum = latest["gas_raw"], latest["temperature_c"], latest["humidity_percent"]
    demo = not live
    cards = [
        readout_card("MQ-3 · raw sensor reading", mq, "ADC", "Raw counts · 12-bit ESP32 ADC", 0, (mq / 4095 * 100) if mq is not None else None, "lime", demo),
        readout_card("Temperature", temp, "°C", "DHT22 sensor" if live else "Simulated", 1, (temp / 50 * 100) if temp is not None else None, "coral", demo),
        readout_card("Relative humidity", hum, "% RH", "DHT22 sensor" if live else "Simulated", 1, hum, "sky", demo),
        readout_card("Total observations", total, "", "Live readings" if live else "Synthetic rows", 0, None, "mustard", demo),
    ]
    for col, card in zip(cols, cards):
        with col:
            _render(card)


# --------------------------------------------------------------------------- #
# AI assessment panel — the big new component
# --------------------------------------------------------------------------- #
def _risk_color_class(level: str) -> str:
    return {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high", "CRITICAL": "risk-critical"}.get(level, "risk-medium")


def _prob_bar_color(label: str) -> str:
    return {"FRESH": "#12B3A4", "RIPENING": "#FFC531", "SPOILED": "#FF5B57"}.get(label, "#6F6A60")


def _alert_class(level: str) -> str:
    return {"CRITICAL": "alert-critical", "URGENT": "alert-urgent", "WARNING": "alert-warning", "CAUTION": "alert-caution"}.get(level, "alert-info")


def render_ai_assessment(specimen_id: str, pred: dict, df: pd.DataFrame) -> None:
    """Full AI assessment panel with risk gauge, shelf-life, confidence, trend, and alerts."""
    label = pred.get("label", "UNKNOWN")
    confidence = pred.get("confidence")
    risk_score = pred.get("risk_score", 0)
    risk_level = pred.get("risk_level", "LOW")
    risk_emoji = pred.get("risk_emoji", "")
    shelf_days = pred.get("shelf_life_days")
    shelf_hours = pred.get("shelf_life_hours")
    probabilities = pred.get("probabilities", {})
    trend = pred.get("trend", {})
    alerts = pred.get("alerts", [])
    model_source = pred.get("model_source", "unknown")

    # Top row: Classification + Confidence
    _render(f"""
    <div class="sw-panel" id="sw-assess">
      <div class="sw-title"><span class="idx">03</span>AI Freshness Assessment
        <span class="tag">{tag(f"{model_source}", "lime", dot=True)}</span>
      </div>
      <div class="sw-assess">
        <div class="big">{risk_emoji} {_e(label)}</div>
        <div class="sub">Confidence: {confidence:.0%} &middot; Risk: {risk_score:.0f}% ({risk_level})</div>
      </div>
    </div>
    """)

    # 4-column layout: Risk Gauge | Shelf Life | Probabilities | Alerts
    c1, c2, c3, c4 = st.columns([1, 1, 1.3, 1.2])

    with c1:
        rc = _risk_color_class(risk_level)
        _render(f"""
        <div class="sw-panel">
          <div class="sw-title"><span class="idx">▲</span>Spoilage Risk</div>
          <div class="risk-gauge">
            <div class="score {rc}">{risk_score:.0f}<small>%</small></div>
            <div class="level {rc}">{risk_emoji} {_e(risk_level)}</div>
          </div>
        </div>
        """)

        # Risk gauge as plotly indicator
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=risk_score,
            number={"suffix": "%", "font": {"size": 28, "color": T.INK}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": T.INK},
                "bar": {"color": "#FF5B57" if risk_score > 70 else "#FFC531" if risk_score > 40 else "#12B3A4"},
                "bgcolor": "#F0EBE0",
                "steps": [
                    {"range": [0, 25], "color": "rgba(18,179,164,0.15)"},
                    {"range": [25, 55], "color": "rgba(255,197,49,0.15)"},
                    {"range": [55, 80], "color": "rgba(255,140,0,0.15)"},
                    {"range": [80, 100], "color": "rgba(255,91,87,0.15)"},
                ],
                "threshold": {"line": {"color": T.INK, "width": 2}, "thickness": 0.8, "value": risk_score},
            },
        ))
        fig.update_layout(
            height=160, margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor=T.PANEL, font=dict(family="DM Sans, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        if shelf_days is not None:
            days_str = f"{shelf_days:.1f}"
            hours_str = f"≈ {shelf_hours:.0f} hours" if shelf_hours else ""
            if shelf_days < 0.5:
                urgency = tag("URGENT", "crimson", dot=True, pulse=True)
            elif shelf_days < 1.5:
                urgency = tag("LOW SHELF LIFE", "amber", dot=True)
            elif shelf_days < 3:
                urgency = tag("MODERATE", "cyan")
            else:
                urgency = tag("GOOD", "lime")
        else:
            days_str = "—"
            hours_str = ""
            urgency = tag("Unavailable", "amber")

        _render(f"""
        <div class="sw-panel">
          <div class="sw-title"><span class="idx">⏱</span>Shelf Life</div>
          <div class="shelf-life">
            <div class="label">Estimated remaining</div>
            <div class="days">{days_str} <small>days</small></div>
            <div class="hours">{hours_str}</div>
            <div style="margin-top:0.5rem">{urgency}</div>
          </div>
        </div>
        """)

    with c3:
        # Probability bars for each class
        prob_bars = ""
        for cls in ["FRESH", "RIPENING", "SPOILED"]:
            p = probabilities.get(cls, 0.0)
            pct = f"{p:.0%}"
            color = _prob_bar_color(cls)
            width = max(2, p * 100)
            prob_bars += f"""
            <div class="prob-bar">
              <span class="bar-label">{cls}</span>
              <div class="bar-track"><div class="bar-fill" style="width:{width:.0f}%;background:{color}"></div></div>
              <span class="bar-pct">{pct}</span>
            </div>
            """

        _render(f"""
        <div class="sw-panel">
          <div class="sw-title"><span class="idx">📊</span>Class Probabilities</div>
          <div style="padding:0.5rem 0">{prob_bars}</div>
          <div class="sw-label" style="margin-top:0.3rem">Confidence: {confidence:.1%}</div>
        </div>
        """)

    with c4:
        # Alerts feed
        if alerts:
            alert_html = ""
            for a in alerts[:5]:
                acls = _alert_class(a["level"])
                alert_html += f'<div class="alert-item {acls}">{a["icon"]} <span>{_e(a["message"])}</span></div>'
        else:
            alert_html = '<div class="sw-label" style="padding:1rem 0;text-align:center">No active alerts</div>'

        _render(f"""
        <div class="sw-panel">
          <div class="sw-title"><span class="idx">🔔</span>Alerts & Warnings</div>
          {alert_html}
        </div>
        """)

    # Trend summary row
    if trend and trend.get("samples", 0) > 1:
        t1, t2, t3, t4 = st.columns(4)
        with t1:
            _render(f"""<div class="sw-panel sw-readout">
              <span class="sw-label">Gas Trend</span>
              <div class="val" style="font-size:1.3rem">{_e(trend.get('gas_trend', '—'))}</div>
              <div class="foot"><span class="note">Rate: {trend.get('gas_rate', 0):+.1f}/reading</span>{tag("Live", "lime", dot=True)}</div>
            </div>""")
        with t2:
            _render(f"""<div class="sw-panel sw-readout">
              <span class="sw-label">Gas Average</span>
              <div class="val" style="font-size:1.3rem">{trend.get('gas_avg', 0):.0f} <small>ADC</small></div>
              <div class="foot"><span class="note">Range: {trend.get('gas_min', 0)}—{trend.get('gas_max', 0)}</span></div>
            </div>""")
        with t3:
            _render(f"""<div class="sw-panel sw-readout">
              <span class="sw-label">Deterioration</span>
              <div class="val" style="font-size:1.3rem">{_e(trend.get('deterioration_speed', '—'))}</div>
              <div class="foot"><span class="note">{trend.get('samples', 0)} samples in window</span></div>
            </div>""")
        with t4:
            _render(f"""<div class="sw-panel sw-readout">
              <span class="sw-label">Environment</span>
              <div class="val" style="font-size:1.3rem">{trend.get('temp_avg', 0):.1f}°C / {trend.get('humidity_avg', 0):.0f}%</div>
              <div class="foot"><span class="note">Temp / Humidity average</span></div>
            </div>""")


# --------------------------------------------------------------------------- #
# Legacy assessment (for demo mode)
# --------------------------------------------------------------------------- #
def render_assessment(specimen_id: str, model_trained: bool, timestamp: str, prediction: Optional[dict] = None, live: bool = False) -> None:
    """Assessment panel for demo/offline mode."""
    body = f"""
    <div class="sw-assess">
      <div class="big">{"MODEL AVAILABLE" if model_trained else "MODEL NOT TRAINED"}</div>
      <div class="sub">{"MODEL OUTPUT: " + _e(prediction["label"]).upper() if prediction else "NO VALIDATED SPOILAGE PREDICTION AVAILABLE"}</div>
    </div>
    <div class="sw-assess-grid">
      <div><span class="sw-label">Assessment status</span><div class="v">{tag("Available", "lime") if prediction else tag("Not available", "amber")}</div></div>
      <div><span class="sw-label">Model availability</span><div class="v">{tag("Available" if model_trained else "Unavailable", "lime" if model_trained else "crimson")}</div></div>
      <div><span class="sw-label">Specimen</span><div class="v">{_e(specimen_id)}</div></div>
      <div><span class="sw-label">Timestamp</span><div class="v">{_e(timestamp)}</div></div>
      <div><span class="sw-label">Prediction</span><div class="v">{_e(prediction["label"]) if prediction else "— — —"}</div></div>
      <div><span class="sw-label">Confidence</span><div class="v">{f'{prediction["confidence"]:.0%}' if prediction and prediction.get("confidence") is not None else "— — —"}</div></div>
    </div>
    <div class="sw-explain">
      <b>Connect ESP32 for live analysis.</b> A trained AI model is loaded and ready.
      Once the ESP32 sensor starts streaming data, you will see real-time classification (FRESH/RIPENING/SPOILED),
      risk scoring, shelf-life estimation, and trend analysis.
    </div>
    <div class="sw-slot"><span class="sw-label">Assessment output slot</span>{tag("Connect sensor for live analysis", "amber")}</div>
    """
    _render(panel("Freshness / spoilage assessment", "03", body, tag("No live data", "amber"), "sw-assess"))


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
def render_observations(df: pd.DataFrame, limit: int = 10, live: bool = False) -> None:
    recent = df.sort_values("timestamp", ascending=False).head(limit)

    # Include spoilage label if present
    has_label = "spoilage_label" in recent.columns

    rows = ""
    for r in recent.itertuples():
        label_cell = ""
        if has_label:
            lbl = getattr(r, "spoilage_label", "")
            tone = "lime" if lbl == "FRESH" else "amber" if lbl == "RIPENING" else "crimson" if lbl == "SPOILED" else ""
            label_cell = f'<td>{tag(str(lbl), tone)}</td>'

        rows += (
            f"<tr><td>{r.timestamp:%Y-%m-%d %H:%M}</td><td>{_e(r.specimen_id)}</td>"
            f'<td class="num">{r.gas_raw:,}</td><td class="num">{r.temperature_c:.1f} °C</td>'
            f'<td class="num">{r.humidity_percent:.1f} %</td>{label_cell}'
            f'<td class="src">{_e(r.source)}</td></tr>'
        )

    label_header = '<th>AI Label</th>' if has_label else ''
    source_tag = tag("Live", "lime") if live else tag("Synthetic", "amber")

    table = (
        f'<div class="sw-scroll"><table class="sw-table"><thead><tr><th>Timestamp</th><th>Specimen</th>'
        f'<th class="num">MQ-3 raw</th><th class="num">Temp</th><th class="num">Humidity</th>{label_header}<th>Source</th></tr></thead>'
        f"<tbody>{rows}</tbody></table></div>"
        f'<div class="sw-label" style="margin-top:.6rem">{"Live sensor readings with AI predictions" if live else "Synthetic demo dataset · fixed seed · generated in memory"}</div>'
    )
    _render(panel("Recent observations", "05", table, source_tag, "sw-obs"))


def _rgba(hex_color: str, a: float = 0.14) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{a})"


def _trace(df: pd.DataFrame, column: str, name: str, color: str, unit: str, live: bool = False) -> go.Figure:
    hover_extra = "LIVE" if live else "SYNTHETIC"
    fig = go.Figure(
        go.Scatter(
            x=df["timestamp"], y=df[column], mode="lines", name=name,
            line=dict(color=color, width=3, shape="spline", smoothing=0.5),
            fill="tozeroy", fillcolor=_rgba(color),
            hovertemplate=f"%{{x|%H:%M}}<br>%{{y:,.1f}} {unit}<extra>{hover_extra}</extra>",
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


def render_signal_trace(df: pd.DataFrame, live: bool = False) -> None:
    with st.container(border=True):
        source_tag = tag("Live sensor data", "lime") if live else tag("Synthetic · not real measurements", "amber")
        _render(
            '<div class="sw-title" style="margin-bottom:.4rem"><span class="idx">06</span>Signal trace'
            f'<span class="tag">{source_tag}</span></div>'
        )
        t1, t2, t3 = st.tabs(["MQ-3 RAW", "TEMPERATURE", "HUMIDITY"])
        cfg = {"displayModeBar": False}
        with t1:
            st.plotly_chart(_trace(df, "gas_raw", "MQ-3 raw", T.TEAL, "ADC counts", live), width="stretch", config=cfg)
        with t2:
            st.plotly_chart(_trace(df, "temperature_c", "Temperature", T.CORAL, "°C", live), width="stretch", config=cfg)
        with t3:
            st.plotly_chart(_trace(df, "humidity_percent", "Humidity", T.SKY, "% RH", live), width="stretch", config=cfg)


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
        "FreshGuard · IoT-based banana spoilage detection · AI-powered classification, shelf-life estimation & risk scoring</div>"
    )
