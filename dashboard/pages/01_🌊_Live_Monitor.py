"""
🌊 Live Monitor — Real-time gauge levels & alert map
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import requests
from dashboard.style import apply_global_css, metric_card

st.set_page_config(page_title="Live Monitor", page_icon="🌊", layout="wide")
apply_global_css()

from datetime import datetime

# Control logic — define before columns to use in both
simulation_date = st.date_input(
    "Simulation Target Date", 
    value=datetime(2022, 7, 15).date(), 
    min_value=datetime(2022, 6, 1).date(), 
    max_value=datetime(2022, 8, 31).date()
)

# Fetch station data
api_url_base = st.session_state.get("api_url", "http://api:8000")
try:
    stations_res = requests.get(
        f"{api_url_base}/gauges/stations", 
        params={"date": simulation_date.strftime("%Y-%m-%d")},
        timeout=5
    )
    if stations_res.status_code == 200:
        raw_stations = stations_res.json().get("stations", [])
        stations_data = pd.DataFrame(raw_stations)
        # Sanitization
        for col, default in [("alert", "GREEN"), ("danger", 15.0), ("warning", 13.0)]:
            if col not in stations_data.columns: stations_data[col] = default
        if "level" not in stations_data.columns:
            stations_data["level"] = stations_data.get("water_level_m", 0)
    else:
        raise ConnectionError(f"API Error {stations_res.status_code}")
except Exception as e:
    st.error(f"Failed to connect to simulation backend: {e}")
    st.stop()

# Alert counts
alert_counts = stations_data["alert"].value_counts()

# ── Layout: Left Control Panel | Right Data Visualization ──
col_controls, col_viz = st.columns([1, 2.5], gap="large")

with col_controls:
    st.markdown("# 🌊 Live Monitor")
    st.markdown("Real-time simulation using GloFAS historical discharge (Jun-Aug 2022)")
    st.info("ℹ️ Monitoring state based on selected date.")
    
    st.markdown("### 📊 Alert Summary")
    # Stack metrics vertically in the control column with compact mode
    metric_card("🔴 Critical Alerts", str(alert_counts.get("RED", 0)), "Exceeding Danger Level", compact=True)
    metric_card("🟠 Warning Alerts", str(alert_counts.get("ORANGE", 0)), "Approaching Danger", compact=True)
    metric_card("🟡 Watch Alerts", str(alert_counts.get("YELLOW", 0)), "Elevated Levels", compact=True)
    metric_card("🟢 Safe Stations", str(alert_counts.get("GREEN", 0)), "Normal Conditions", compact=True)

with col_viz:
    tab_map, tab_table = st.tabs(["📍 Interactive Map", "📋 Station Table"])
    
    with tab_map:
        ALERT_COLORS = {"GREEN": "green", "YELLOW": "orange", "ORANGE": "red", "RED": "darkred"}
        m = folium.Map(location=[22.0, 82.0], zoom_start=5, tiles="cartodbdark_matter")

        for _, row in stations_data.iterrows():
            color = ALERT_COLORS.get(row["alert"], "gray")
            popup_html = f"""
            <div style="font-family: Arial; min-width: 180px;">
                <h4 style="margin-bottom:5px;">{row['name']}</h4>
                <b>River:</b> {row['river']}<br>
                <b>Level:</b> {row['level']:.1f} m<br>
                <b>Status:</b> <span style="color:{color};">{row['alert']}</span>
            </div>
            """
            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=10 if row["alert"] in ["RED", "ORANGE"] else 6,
                popup=folium.Popup(popup_html, max_width=300),
                color=color, fill=True, fill_opacity=0.7, weight=2,
            ).add_to(m)

        st_folium(m, width="100%", height=600)

    with tab_table:
        st.markdown("### Detailed Conditions")
        # List view with progress-style indicators
        for _, row in stations_data.sort_values("alert", key=lambda x: x.map(
            {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
        )).iterrows():
            alert_class = f"alert-{row['alert'].lower()}"
            pct = min(100, row["level"] / row["danger"] * 100)
            
            st.markdown(f"""
            <div class="{alert_class}" style="margin-bottom: 12px; border-radius: 8px; padding: 15px;">
                <div style="display: flex; justify-content: space-between;">
                    <b>{row['name']}</b>
                    <span>{row['level']:.1f}m / {row['danger']:.1f}m</span>
                </div>
                <div style="background: rgba(0,0,0,0.2); height: 4px; border-radius: 2px; margin-top: 8px;">
                    <div style="background: white; width: {pct}%; height: 100%; border-radius: 2px;"></div>
                </div>
                <small style="opacity: 0.8">{row['river']} Basin</small>
            </div>
            """, unsafe_allow_html=True)
