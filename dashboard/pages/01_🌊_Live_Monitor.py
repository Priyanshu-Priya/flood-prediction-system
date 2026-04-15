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

st.markdown("# 🌊 Live Flood Monitor")
st.markdown("Real-time water levels from GloFAS virtual telemetry stations across the continent")
st.markdown("<br>", unsafe_allow_html=True)

# Fetch station data from API (fallback to sample data)
api_url_base = "http://api:8000" if "api_url" not in st.session_state else st.session_state["api_url"]
api_url = api_url_base

try:
    stations_res = requests.get(f"{api_url}/gauges/stations", timeout=3)
    if stations_res.status_code == 200:
        raw_stations = stations_res.json().get("stations", [])
        if raw_stations:
            stations_data = pd.DataFrame(raw_stations)
            # Ensure required columns with defaults
            for col, default in [("alert", "GREEN"), ("danger", 15.0), ("warning", 13.0)]:
                if col not in stations_data.columns:
                    stations_data[col] = default
            if "level" not in stations_data.columns:
                stations_data["level"] = stations_data.get("water_level_m", 0)
            if "danger" not in stations_data.columns:
                stations_data["danger"] = stations_data.get("danger_level_m", 15.0)
        else:
            raise ValueError("Empty station list")
    else:
        raise ConnectionError(f"API returned {stations_res.status_code}")
except Exception:
    # Fallback sample data
    stations_data = pd.DataFrame([
        {"name": "Farakka Barrage (Synthetic)", "lat": 24.81, "lon": 87.92, "river": "Ganga",
         "level": 12.5, "danger": 15.0, "warning": 13.5, "alert": "GREEN"},
        {"name": "Varanasi (Synthetic)", "lat": 25.32, "lon": 83.01, "river": "Ganga",
         "level": 14.2, "danger": 15.0, "warning": 13.5, "alert": "ORANGE"},
        {"name": "Patna (Synthetic)", "lat": 25.60, "lon": 85.10, "river": "Ganga",
         "level": 10.8, "danger": 12.0, "warning": 10.5, "alert": "YELLOW"},
        {"name": "Dibrugarh (Synthetic)", "lat": 27.47, "lon": 94.91, "river": "Brahmaputra",
         "level": 16.3, "danger": 16.0, "warning": 14.5, "alert": "RED"},
    ])

# Alert summary
col1, col2, col3, col4 = st.columns(4)
alert_counts = stations_data["alert"].value_counts()
with col1:
    metric_card("🔴 Critical Alerts", str(alert_counts.get("RED", 0)), "Exceeding Danger Level")
with col2:
    metric_card("🟠 Warning Alerts", str(alert_counts.get("ORANGE", 0)), "Approaching Danger")
with col3:
    metric_card("🟡 Watch Alerts", str(alert_counts.get("YELLOW", 0)), "Elevated Levels")
with col4:
    metric_card("🟢 Safe Stations", str(alert_counts.get("GREEN", 0)), "Normal Conditions")

st.markdown("---")

# Map
col_map, col_table = st.columns([2, 1])

with col_map:
    st.markdown("### 📍 Station Map")

    ALERT_COLORS = {"GREEN": "green", "YELLOW": "orange", "ORANGE": "red", "RED": "darkred"}

    m = folium.Map(
        location=[22.0, 82.0],
        zoom_start=5,
        tiles="cartodbdark_matter",
    )

    for _, row in stations_data.iterrows():
        color = ALERT_COLORS.get(row["alert"], "gray")
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4>{row['name']}</h4>
            <b>River:</b> {row['river']}<br>
            <b>Level:</b> {row['level']:.1f} m<br>
            <b>Warning:</b> {row['warning']:.1f} m<br>
            <b>Danger:</b> {row['danger']:.1f} m<br>
            <b>Alert:</b> <span style="color:{color};font-weight:bold">{row['alert']}</span>
        </div>
        """
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=12 if row["alert"] in ["RED", "ORANGE"] else 8,
            popup=folium.Popup(popup_html, max_width=300),
            color=color,
            fill=True,
            fill_opacity=0.8,
            weight=2,
        ).add_to(m)

    st_folium(m, width=700, height=500)

with col_table:
    st.markdown("### 📋 Station Status")

    # Show alerts with color coding
    for _, row in stations_data.sort_values("alert", key=lambda x: x.map(
        {"RED": 0, "ORANGE": 1, "YELLOW": 2, "GREEN": 3}
    )).iterrows():
        alert_class = f"alert-{row['alert'].lower()}"
        pct = row["level"] / row["danger"] * 100

        st.markdown(f"""
        <div class="{alert_class}" style="margin-bottom: 8px; border-radius: 8px; padding: 10px;">
            <b>{row['name']}</b> ({row['river']})<br>
            Level: {row['level']:.1f}m / {row['danger']:.1f}m ({pct:.0f}%)
        </div>
        """, unsafe_allow_html=True)
