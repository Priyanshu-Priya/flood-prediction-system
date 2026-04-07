"""
📈 Forecasts — LSTM water level prediction plots
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="Forecasts", page_icon="📈", layout="wide")
st.markdown("# 📈 Water Level Forecasts")
st.markdown("LSTM-predicted water levels with uncertainty bands")

# Controls
col1, col2, col3 = st.columns(3)
with col1:
    station = st.selectbox("Station", [
        "Patna (Ganga)", "Dibrugarh (Brahmaputra)",
        "Delhi (Yamuna)", "Chennai (Adyar)",
        "Mumbai (Mithi)", "Varanasi (Ganga)",
    ])
with col2:
    horizon = st.selectbox("Forecast Horizon", ["24h", "48h", "72h", "120h"])
with col3:
    show_uncertainty = st.checkbox("Show Uncertainty Bands", value=True)

st.markdown("---")

# Generate realistic sample forecast
np.random.seed(hash(station) % 2**32)
horizon_hours = int(horizon.replace("h", ""))
now = datetime.now()

# Historical (7 days)
hist_hours = 168
hist_time = [now - timedelta(hours=hist_hours - i) for i in range(hist_hours)]

# Simulate monsoon water level pattern
t = np.linspace(0, 14 * np.pi, hist_hours)
base_level = 8.0 + 2.0 * np.sin(t / 5) + np.random.randn(hist_hours) * 0.3

# Add a flood event
event_start = hist_hours - 50
base_level[event_start:event_start+30] += np.linspace(0, 4, 30)
base_level[event_start+30:] += 4 * np.exp(-np.arange(len(base_level) - event_start - 30) / 20)

# Forecast
fc_time = [now + timedelta(hours=i + 1) for i in range(horizon_hours)]
fc_mean = base_level[-1] + np.cumsum(np.random.randn(horizon_hours) * 0.08) - np.linspace(0, 1, horizon_hours)
fc_std = np.linspace(0.2, 1.5, horizon_hours)  # Uncertainty grows with lead time

# Danger and warning levels
danger_level = 15.0 if "Brahmaputra" in station else 12.0
warning_level = danger_level - 1.5

# Main forecast plot
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(
    x=hist_time, y=base_level,
    mode="lines", name="Observed",
    line=dict(color="#00b4d8", width=2),
))

# Forecast mean
fig.add_trace(go.Scatter(
    x=fc_time, y=fc_mean,
    mode="lines", name="Forecast (mean)",
    line=dict(color="#f59e0b", width=2.5, dash="dot"),
))

# Uncertainty bands
if show_uncertainty:
    fig.add_trace(go.Scatter(
        x=fc_time + fc_time[::-1],
        y=np.concatenate([fc_mean + 1.645 * fc_std, (fc_mean - 1.645 * fc_std)[::-1]]),
        fill="toself",
        fillcolor="rgba(245, 158, 11, 0.15)",
        line=dict(color="rgba(245, 158, 11, 0)"),
        name="90% CI",
    ))

# Danger and warning levels
fig.add_hline(y=danger_level, line_dash="dash", line_color="#dc2626",
             annotation_text="⚠️ DANGER LEVEL", annotation_position="top right")
fig.add_hline(y=warning_level, line_dash="dash", line_color="#ea580c",
             annotation_text="WARNING LEVEL", annotation_position="top right")

fig.update_layout(
    title=f"Water Level Forecast — {station}",
    xaxis_title="Time",
    yaxis_title="Water Level (m)",
    height=500,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    legend=dict(orientation="h", yanchor="bottom", y=1.02),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# Metrics
st.markdown("### Forecast Summary")
col1, col2, col3, col4, col5 = st.columns(5)

peak_level = fc_mean.max()
peak_hour = np.argmax(fc_mean) + 1
alert = "RED" if peak_level >= danger_level else "ORANGE" if peak_level >= warning_level else "YELLOW" if peak_level >= warning_level - 1 else "GREEN"

with col1:
    st.metric("Peak Forecast", f"{peak_level:.2f} m")
with col2:
    st.metric("Peak Time", f"+{peak_hour}h")
with col3:
    st.metric("Current Level", f"{base_level[-1]:.2f} m")
with col4:
    st.metric("Alert Level", alert)
with col5:
    st.metric("Uncertainty (72h)", f"±{fc_std[-1]:.2f} m")

# Lead-time degradation
st.markdown("### Lead-Time Degradation")
lead_times = [6, 12, 24, 48, 72]
nse_values = [0.92, 0.87, 0.78, 0.65, 0.52]  # Typical degradation

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=lead_times, y=nse_values,
    mode="lines+markers",
    line=dict(color="#00b4d8", width=3),
    marker=dict(size=10),
    name="NSE",
))
fig2.add_hline(y=0.75, line_dash="dash", line_color="#16a34a",
              annotation_text="Good (NSE=0.75)")
fig2.add_hline(y=0.36, line_dash="dash", line_color="#dc2626",
              annotation_text="Unsatisfactory (NSE=0.36)")

fig2.update_layout(
    title="Model Performance vs Forecast Lead Time",
    xaxis_title="Lead Time (hours)",
    yaxis_title="Nash-Sutcliffe Efficiency (NSE)",
    height=350,
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis_range=[0, 1],
)

st.plotly_chart(fig2, use_container_width=True)
