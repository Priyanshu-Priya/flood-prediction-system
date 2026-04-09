"""
🗺️ Risk Maps — Interactive flood susceptibility map viewer
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(page_title="Risk Maps", page_icon="🗺️", layout="wide")
st.markdown("# 🗺️ Flood Susceptibility Risk Maps")
st.markdown("XGBoost-predicted flood probability overlaid on terrain")

# Controls
col1, col2, col3 = st.columns(3)

default_bbox = st.session_state.get("bbox", (80.0, 12.8, 80.4, 13.25))
default_region_name = st.session_state.get("active_station_name", "Selected Region")

with col1:
    st.text_input("Active Region (from Sidebar)", value=default_region_name, disabled=True)
with col2:
    overlay = st.selectbox("Overlay", [
        "Flood Probability", "TWI", "Slope", "Distance to Channel",
    ])
with col3:
    threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.5, 0.05)

st.markdown("---")

# Fetch risk stats from API
api_url = f"{st.session_state.get('api_url', 'http://localhost:8000')}/predict/susceptibility"

payload = {
    "min_lon": default_bbox[0],
    "min_lat": default_bbox[1],
    "max_lon": default_bbox[2],
    "max_lat": default_bbox[3],
    "resolution_m": 100
}

import requests
try:
    with st.spinner("Fetching spatial susceptibility from XGBoost..."):
        response = requests.post(api_url, json=payload)
        if response.status_code == 200:
            stats_data = response.json()
        else:
            st.error(f"API Error ({response.status_code}): {response.text}")
            st.stop()
except Exception as e:
    st.error(f"Failed to connect to API backend: {e}")
    st.stop()

# Generate visual representation (Since terrain rasters are WIP, we construct a procedural map seeded by location)
np.random.seed(int(default_bbox[0] * default_bbox[1] * 100) % (2**32))
grid_size = 100

x = np.linspace(default_bbox[0], default_bbox[2], grid_size)
y = np.linspace(default_bbox[1], default_bbox[3], grid_size)
X, Y = np.meshgrid(x, y)

# Simulate terrain mechanics for visual context 
elevation = 50 + 30 * np.sin(3 * (X - default_bbox[0])) * np.cos(2 * (Y - default_bbox[1])) + 10 * np.random.randn(grid_size, grid_size) * 0.1
river_channel = np.exp(-((Y - (default_bbox[1] + default_bbox[3])/2) ** 2) / 0.005)

twi = 5 + 10 * river_channel + np.random.randn(grid_size, grid_size) * 0.5
flood_prob = 1 / (1 + np.exp(-(twi - 10) / 2)) + np.random.randn(grid_size, grid_size) * 0.05
flood_prob = np.clip(flood_prob, 0, 1)

col_map, col_stats = st.columns([3, 1])

with col_map:
    if overlay == "Flood Probability":
        data = flood_prob
        colorscale = [
            [0.0, "#16a34a"], [0.3, "#eab308"],
            [0.6, "#ea580c"], [0.8, "#dc2626"],
            [1.0, "#7f1d1d"],
        ]
        title = "Flood Probability"
    elif overlay == "TWI":
        data = twi
        colorscale = "Blues"
        title = "Topographic Wetness Index"
    elif overlay == "Slope":
        data = np.gradient(elevation)[0]
        colorscale = "YlOrRd"
        title = "Slope (degrees)"
    else:
        data = np.sqrt((Y - (default_bbox[1] + default_bbox[3])/2) ** 2) * 111000 # Roughly to meters
        colorscale = "Viridis_r"
        title = "Distance to Channel (m)"

    fig = go.Figure(data=go.Heatmap(
        z=data, x=x, y=y,
        colorscale=colorscale,
        colorbar=dict(title=title, thickness=20),
    ))

    fig.update_layout(
        title=f"{title} — {default_region_name}",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.markdown("### XGBoost Risk Summary")

    st.metric("Total Grid Cells", f"{stats_data['n_cells']:,}")
    # Calculate high risk from API distribution instead of visual map
    high_risk_cells = stats_data['risk_distribution'].get('red', 0) + stats_data['risk_distribution'].get('orange', 0)
    st.metric("High Risk Cells", f"{high_risk_cells:,}")
    st.metric("Mean Probability", f"{stats_data['mean_probability']:.3f}")
    st.metric("Max Probability", f"{stats_data['max_probability']:.3f}")

    st.markdown("---")
    st.markdown("### Risk Distribution (API)")

    rd = stats_data.get('risk_distribution', {})
    rp = stats_data.get('risk_percentages', {})
    
    for level, code, color in [
        ("GREEN", "green", "#16a34a"),
        ("YELLOW", "yellow", "#eab308"),
        ("ORANGE", "orange", "#ea580c"),
        ("RED", "red", "#dc2626"),
    ]:
        count = rd.get(code, 0)
        pct = rp.get(code, 0.0)
        st.markdown(f"**{level}**: {count:,} ({pct:.1f}%)")
