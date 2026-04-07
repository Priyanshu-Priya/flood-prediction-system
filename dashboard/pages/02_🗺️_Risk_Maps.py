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

with col1:
    region = st.selectbox("Region", [
        "Greater Chennai", "Mumbai Metropolitan",
        "Brahmaputra Basin", "Ganga Basin (Bihar)",
    ])
with col2:
    overlay = st.selectbox("Overlay", [
        "Flood Probability", "TWI", "Slope", "Distance to Channel",
    ])
with col3:
    threshold = st.slider("Alert Threshold", 0.0, 1.0, 0.5, 0.05)

st.markdown("---")

# Generate sample data for visualization
np.random.seed(42)
grid_size = 100

# Create synthetic but realistic-looking terrain
x = np.linspace(0, 1, grid_size)
y = np.linspace(0, 1, grid_size)
X, Y = np.meshgrid(x, y)

# Simulate a river valley with flood plain
elevation = 50 + 30 * np.sin(3 * X) * np.cos(2 * Y) + 10 * np.random.randn(grid_size, grid_size) * 0.1
river_channel = np.exp(-((Y - 0.5) ** 2) / 0.01)

# TWI (high near river, low on slopes)
twi = 5 + 10 * river_channel + np.random.randn(grid_size, grid_size) * 0.5

# Flood probability (correlated with TWI)
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
        data = np.sqrt((Y - 0.5) ** 2) * 1000
        colorscale = "Viridis_r"
        title = "Distance to Channel (m)"

    fig = go.Figure(data=go.Heatmap(
        z=data,
        colorscale=colorscale,
        colorbar=dict(title=title, thickness=20),
    ))

    fig.update_layout(
        title=f"{title} — {region}",
        xaxis_title="Easting (grid cells)",
        yaxis_title="Northing (grid cells)",
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )

    st.plotly_chart(fig, use_container_width=True)

with col_stats:
    st.markdown("### Risk Summary")

    total_cells = grid_size ** 2
    high_risk = (flood_prob >= threshold).sum()

    st.metric("Total Grid Cells", f"{total_cells:,}")
    st.metric("High Risk Cells", f"{high_risk:,}", delta=f"{high_risk/total_cells*100:.1f}%")
    st.metric("Mean Probability", f"{flood_prob.mean():.3f}")
    st.metric("Max Probability", f"{flood_prob.max():.3f}")
    st.metric("P95 Probability", f"{np.percentile(flood_prob, 95):.3f}")

    st.markdown("---")
    st.markdown("### Risk Distribution")

    for level, (low, high), color in [
        ("🟢 GREEN", (0, 0.3), "#16a34a"),
        ("🟡 YELLOW", (0.3, 0.6), "#eab308"),
        ("🟠 ORANGE", (0.6, 0.8), "#ea580c"),
        ("🔴 RED", (0.8, 1.0), "#dc2626"),
    ]:
        count = ((flood_prob >= low) & (flood_prob < high)).sum()
        pct = count / total_cells * 100
        st.markdown(f"**{level}**: {count:,} ({pct:.1f}%)")
