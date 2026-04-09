"""
🌊 Flood Risk Prediction Dashboard — India
=============================================
Multi-page Streamlit dashboard for real-time flood monitoring,
risk map visualization, LSTM forecast analysis, and model analytics.

Run: streamlit run dashboard/app.py --server.port 8501
"""

import streamlit as st
import requests
import os

# ── API Configuration ──
DEFAULT_API_URL = os.getenv("FLOOD_DASH_api_base_url", "http://localhost:8000")
if "api_url" not in st.session_state:
    st.session_state["api_url"] = DEFAULT_API_URL

# ── Page Configuration ──
st.set_page_config(
    page_title="🌊 Flood Risk Prediction — India",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown("""
<style>
    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 50%, #0d1117 100%);
    }

    .main-header {
        background: linear-gradient(90deg, #00b4d8, #0077b6, #023e8a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.2rem;
        backdrop-filter: blur(10px);
        transition: transform 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #00b4d8;
    }

    .alert-red {
        background: linear-gradient(135deg, #dc2626, #991b1b);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        font-weight: 600;
    }

    .alert-orange {
        background: linear-gradient(135deg, #ea580c, #c2410c);
        color: white;
        padding: 1rem;
        border-radius: 8px;
    }

    .alert-yellow {
        background: linear-gradient(135deg, #eab308, #ca8a04);
        color: #1a1a1a;
        padding: 1rem;
        border-radius: 8px;
    }

    .alert-green {
        background: linear-gradient(135deg, #16a34a, #15803d);
        color: white;
        padding: 1rem;
        border-radius: 8px;
    }

    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 10px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──
with st.sidebar:
    st.markdown("## 🌊 Flood Risk India")
    st.markdown("---")

    # API Status
    api_url = st.text_input(
        "API URL",
        value=st.session_state["api_url"],
        key="api_url_input",
        help="FastAPI prediction service URL (use 'http://api:8000' inside Docker)",
    )
    st.session_state["api_url"] = api_url

    st.markdown("---")

    # AOI Selection
    st.markdown("### 📍 Area of Interest")
    aoi_preset = st.selectbox(
        "Quick Select",
        [
            "Global Search",
            "Custom",
            "Greater Chennai",
            "Mumbai Metropolitan",
            "Brahmaputra Basin (Assam)",
            "Ganga Basin (Bihar)",
            "Kerala Coast",
            "Delhi-NCR / Yamuna",
            "Bengaluru Urban",
        ],
    )

    # Preset bounding boxes for Indian flood-prone regions
    PRESETS = {
        "Greater Chennai": (80.0, 12.8, 80.4, 13.25),
        "Mumbai Metropolitan": (72.7, 18.85, 73.1, 19.3),
        "Brahmaputra Basin (Assam)": (89.5, 25.5, 96.0, 28.0),
        "Ganga Basin (Bihar)": (83.5, 24.5, 87.5, 27.5),
        "Kerala Coast": (75.5, 8.0, 77.5, 12.5),
        "Delhi-NCR / Yamuna": (76.8, 28.3, 77.5, 28.9),
        "Bengaluru Urban": (77.4, 12.8, 77.8, 13.15),
    }

    if aoi_preset == "Global Search":
        search_query = st.text_input("Enter city or coordinates", placeholder="e.g. Paris, France")
        if search_query:
            import geopy
            from geopy.geocoders import Nominatim
            try:
                geolocator = Nominatim(user_agent="flood_risk_dashboard")
                location = geolocator.geocode(search_query)
                if location:
                    st.success(f"📍 {location.address}")
                    bbox_str = location.raw.get('boundingbox')
                    if bbox_str:
                        # Nominatim bbox: [lat_min, lat_max, lon_min, lon_max]
                        min_lat, max_lat, min_lon, max_lon = map(float, bbox_str)
                    else:
                        min_lat = location.latitude - 0.1
                        max_lat = location.latitude + 0.1
                        min_lon = location.longitude - 0.1
                        max_lon = location.longitude + 0.1
                    
                    st.session_state["bbox"] = (min_lon, min_lat, max_lon, max_lat)
                    st.session_state["search_lat"] = location.latitude
                    st.session_state["search_lon"] = location.longitude
                    st.session_state["active_station_name"] = search_query.split(',')[0].title()
                else:
                    st.error("Location not found.")
            except Exception as e:
                st.error("Geocoding service unavailable.")
    elif aoi_preset == "Custom":
        col1, col2 = st.columns(2)
        with col1:
            min_lon = st.number_input("Min Lon", value=80.0, step=0.1)
            min_lat = st.number_input("Min Lat", value=12.8, step=0.1)
        with col2:
            max_lon = st.number_input("Max Lon", value=80.4, step=0.1)
            max_lat = st.number_input("Max Lat", value=13.2, step=0.1)
        st.session_state["bbox"] = (min_lon, min_lat, max_lon, max_lat)
        st.session_state["active_station_name"] = "Custom Location"
    else:
        st.session_state["bbox"] = PRESETS[aoi_preset]
        st.session_state["active_station_name"] = aoi_preset

    st.markdown("---")
    st.markdown(
        "**Built with** PyTorch + XGBoost + FastAPI\n\n"
        "**Data**: GloFAS | IMD | Sentinel-1 | ALOS PALSAR"
    )


# ── Main Page ──
st.markdown('<h1 class="main-header">🌊 Flood Risk Prediction System</h1>', unsafe_allow_html=True)
st.markdown("##### Real-time flood monitoring & prediction for India")

# Quick stats — fetched from API
api_url = st.session_state["api_url"]

try:
    health = requests.get(f"{api_url}/health", timeout=2).json()
    metrics_res = requests.get(f"{api_url}/predict/metrics", timeout=3).json()
    n_stations = health.get("stations_registered", 10)
    nse_val = metrics_res.get("lstm", {}).get("nse_mean", 0.0)
    nse_display = f"{nse_val:.2f}" if nse_val > 0 else "—"
    data_source = metrics_res.get("lstm", {}).get("data_source", "none")
    source_label = f"({data_source})" if data_source != "none" else "(not trained)"
except Exception:
    n_stations = 10
    nse_display = "—"
    source_label = "(API offline)"

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("🌧️ Active Alerts", "—", help="Connect API for live alerts")
with col2:
    st.metric("📊 Monitored Stations", str(n_stations), help="GloFAS virtual telemetry stations")
with col3:
    st.metric("🎯 Model NSE", nse_display, help=f"Nash-Sutcliffe Efficiency {source_label}")
with col4:
    st.metric("🗺️ Coverage", "India-wide", help="All major river basins")

st.markdown("---")

# Navigation cards
st.markdown("### Navigate to")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
        <h3>🌊 Live Monitor</h3>
        <p>Real-time gauge levels with alert map</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Live Monitor", key="nav_live", use_container_width=True):
        st.switch_page("pages/01_🌊_Live_Monitor.py")

with c2:
    st.markdown("""
    <div class="metric-card">
        <h3>🗺️ Risk Maps</h3>
        <p>Interactive flood susceptibility viewer</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Risk Maps", key="nav_risk", use_container_width=True):
        st.switch_page("pages/02_🗺️_Risk_Maps.py")

with c3:
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Forecasts</h3>
        <p>LSTM water level predictions</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Forecasts", key="nav_forecast", use_container_width=True):
        st.switch_page("pages/03_📈_Forecasts.py")

with c4:
    st.markdown("""
    <div class="metric-card">
        <h3>📊 Analytics</h3>
        <p>Model performance & history</p>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Go to Analytics", key="nav_analytics", use_container_width=True):
        st.switch_page("pages/04_📊_Analytics.py")

st.markdown("---")

# System Architecture diagram
with st.expander("🏗️ System Architecture", expanded=False):
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────┐
    │                  DATA INGESTION LAYER                    │
    │  Sentinel-1 SAR │ GloFAS EWDS│ IMD Rainfall │ SMAP     │
    │  (STAC API)     │ (Global)  │ (GPM/ERA5)   │ (Soil)   │
    └────────┬────────┴─────┬─────┴──────┬───────┴──────┬───┘
             │              │            │              │
    ┌────────▼──────────────▼────────────▼──────────────▼───┐
    │               FEATURE ENGINEERING                      │
    │  TWI │ Slope │ Flow Accum │ API │ SAR Water │ LULC    │
    └──────────────┬─────────────────────┬──────────────────┘
                   │                     │
    ┌──────────────▼───┐   ┌─────────────▼──────────────────┐
    │   LSTM (PyTorch)  │   │    XGBoost (GPU-accelerated)   │
    │   Water Level     │   │    Spatial Susceptibility       │
    │   Forecasting     │   │    Mapping                      │
    └────────┬──────────┘   └────────────┬───────────────────┘
             │                           │
    ┌────────▼───────────────────────────▼───────────────────┐
    │            ENSEMBLE COMBINER (α-weighted)               │
    │       P_flood = α·P_temporal + (1-α)·P_spatial         │
    └──────────────────────┬─────────────────────────────────┘
                           │
    ┌──────────────────────▼─────────────────────────────────┐
    │     FastAPI (REST + WebSocket) → Streamlit Dashboard    │
    │     Docker Compose: api + dashboard + training          │
    └────────────────────────────────────────────────────────┘
    ```
    """)
