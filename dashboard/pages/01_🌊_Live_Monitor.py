"""
🌊 Live Monitor — Real-time gauge levels & alert map
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np

st.set_page_config(page_title="Live Monitor", page_icon="🌊", layout="wide")
st.markdown("# 🌊 Live Flood Monitor")
st.markdown("Real-time water levels from GloFAS virtual telemetry stations across India")

# Sample station data (production would pull from API)
stations_data = pd.DataFrame([
    {"name": "Farakka Barrage", "lat": 24.81, "lon": 87.92, "river": "Ganga",
     "level": 12.5, "danger": 15.0, "warning": 13.5, "alert": "GREEN"},
    {"name": "Varanasi", "lat": 25.32, "lon": 83.01, "river": "Ganga",
     "level": 14.2, "danger": 15.0, "warning": 13.5, "alert": "ORANGE"},
    {"name": "Patna", "lat": 25.60, "lon": 85.10, "river": "Ganga",
     "level": 10.8, "danger": 12.0, "warning": 10.5, "alert": "YELLOW"},
    {"name": "Dibrugarh", "lat": 27.47, "lon": 94.91, "river": "Brahmaputra",
     "level": 16.3, "danger": 16.0, "warning": 14.5, "alert": "RED"},
    {"name": "Guwahati", "lat": 26.19, "lon": 91.75, "river": "Brahmaputra",
     "level": 11.5, "danger": 14.0, "warning": 12.5, "alert": "GREEN"},
    {"name": "Delhi Yamuna", "lat": 28.68, "lon": 77.24, "river": "Yamuna",
     "level": 7.2, "danger": 8.5, "warning": 7.5, "alert": "GREEN"},
    {"name": "Mumbai Mithi", "lat": 19.07, "lon": 72.88, "river": "Mithi",
     "level": 3.8, "danger": 4.0, "warning": 3.5, "alert": "ORANGE"},
    {"name": "Adyar Chennai", "lat": 13.01, "lon": 80.25, "river": "Adyar",
     "level": 2.1, "danger": 3.5, "warning": 2.8, "alert": "GREEN"},
    {"name": "Surat Tapi", "lat": 21.17, "lon": 72.83, "river": "Tapi",
     "level": 9.2, "danger": 11.0, "warning": 9.5, "alert": "GREEN"},
    {"name": "Vijayawada", "lat": 16.52, "lon": 80.62, "river": "Krishna",
     "level": 8.7, "danger": 10.0, "warning": 9.0, "alert": "GREEN"},
])

# Alert summary
col1, col2, col3, col4 = st.columns(4)
alert_counts = stations_data["alert"].value_counts()
with col1:
    st.metric("🔴 RED", alert_counts.get("RED", 0))
with col2:
    st.metric("🟠 ORANGE", alert_counts.get("ORANGE", 0))
with col3:
    st.metric("🟡 YELLOW", alert_counts.get("YELLOW", 0))
with col4:
    st.metric("🟢 GREEN", alert_counts.get("GREEN", 0))

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
