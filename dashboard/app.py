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

    # 📊 Latest Training Report
    try:
        metrics_res = requests.get(f"{st.session_state['api_url']}/predict/metrics", timeout=2).json()
        lstm_metrics = metrics_res.get("lstm", {})
        
        if lstm_metrics.get("nse_mean", 0) > 0:
            with st.sidebar.expander("📊 Latest Training Report", expanded=True):
                st.markdown(f"**Model**: Pan-India LSTM")
                st.success(f"**Accuracy (NSE)**: {lstm_metrics.get('nse_mean', 0):.4f}")
                st.info(f"**Coverage**: 10 Gauge Stations")
                st.markdown(f"*Last Run: {lstm_metrics.get('last_train', 'N/A')}*")
                if st.button("View Full Report", key="view_full_report", use_container_width=True):
                    st.switch_page("pages/04_📊_Analytics.py")
    except:
        pass

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

# System Architecture
with st.expander("🏗️ System Architecture", expanded=False):
    tab_ml, tab_data, tab_deploy = st.tabs([
        "🧠 ML Pipeline",
        "📡 Data & Features",
        "🐳 Deployment",
    ])

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 1 — Machine Learning Pipeline (primary)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_ml:
        st.markdown("#### End-to-End ML Architecture")
        st.markdown("""
        ```
        ┌────────────────────────────────────────┐
        │          Raw Hydro-Meteorological       │
        │              Time-Series Data           │
        │   (discharge, rainfall, soil moisture)  │
        └───────────────┬────────────────────────┘
                        │
                        ▼
        ┌────────────────────────────────────────┐
        │  ① LSTM Forecaster  (PyTorch)          │──── "When will it flood?"
        │                                         │
        │  Input Projection → LayerNorm → ReLU    │
        │         │                                │
        │  Hindcast LSTM (2 layers, 128-d)        │
        │         │                                │
        │  Temporal Attention (4 heads)            │
        │         │                                │
        │  Forecast LSTM (context + future met)   │
        │         │                                │
        │  Output: μ (mean)  +  σ (uncertainty)   │
        │  Loss:  Gaussian NLL (probabilistic)    │
        └──────────────┬─────────────────────────┘
                       │  P_temporal
                       │
        ┌──────────────▼─────────────────────────┐
        │         ③ Ensemble Combiner             │
        │                                         │
        │  P_flood = α·P_temporal + (1-α)·P_spatial│
        │  α calibrated per basin (Brier Skill)   │
        │                                         │
        │  Risk levels:  🟢 < 0.3 │ 🟡 0.3–0.6   │
        │                🟠 0.6–0.8│ 🔴 ≥ 0.8     │
        └────────────────────────────────────────┘
                       ▲  P_spatial
                       │
        ┌──────────────┴─────────────────────────┐
        │  ② XGBoost Classifier (GPU)            │──── "Where will it flood?"
        │                                         │
        │  15 features per grid cell              │
        │  Objective: binary:logistic             │
        │  HPO: Optuna Bayesian (50 trials)       │
        │  Output: Flood susceptibility map       │
        └────────────────────────────────────────┘
        ```
        """)

        st.markdown("---")

        # LSTM Details
        st.markdown("##### ① Hindcast-Forecast LSTM — Temporal Model")
        col_lstm1, col_lstm2 = st.columns(2)
        with col_lstm1:
            st.markdown("""
            **Architecture**
            - Dual LSTM with **Temporal Attention**
            - Hidden dim: `128` · Layers: `2` · Heads: `4`
            - Lookback: **168 steps** (7 days hourly)
            - Forecast: **72 steps** (3 days ahead)
            - Probabilistic output (μ ± σ → 90% CI)
            """)
        with col_lstm2:
            st.markdown("""
            **Training Pipeline**
            - Walk-forward temporal CV (no data leakage)
            - Mixed-precision (FP16 via `torch.cuda.amp`)
            - AdamW + Cosine Annealing LR
            - Gradient clipping · Early stopping
            - Bayesian HPO via Optuna
            """)

        st.markdown("---")

        # XGBoost Details
        st.markdown("##### ② XGBoost — Spatial Susceptibility Model")
        col_xgb1, col_xgb2 = st.columns(2)
        with col_xgb1:
            st.markdown("""
            | Feature | Source | Type |
            |---------|--------|------|
            | TWI (Topographic Wetness) | DEM | Static |
            | Slope / Aspect | DEM | Static |
            | Flow Accumulation | DEM | Static |
            | Dist. to Channel | DEM | Static |
            | Curvature | DEM | Static |
            | Elevation | DEM | Static |
            """)
        with col_xgb2:
            st.markdown("""
            | Feature | Source | Type |
            |---------|--------|------|
            | LULC Class | ESA WorldCover | Semi-static |
            | Impervious % | WorldCover | Semi-static |
            | Runoff Coefficient | WorldCover | Semi-static |
            | SAR Water Freq. | Sentinel-1 | Dynamic |
            | Hist. Flood Count | SAR Archive | Static |

            **GPU**: `tree_method=gpu_hist` · CUDA 12.x
            **Validation**: Spatial CV (leave-one-watershed-out)
            """)

        st.markdown("---")

        # Ensemble
        st.markdown("##### ③ Ensemble Combiner")
        st.markdown("""
        Fuses the **temporal** (LSTM → *"when"*) and **spatial** (XGBoost → *"where"*) predictions:

        > **`P_flood(x, y, t) = α × P_temporal(t) + (1 − α) × P_spatial(x, y)`**

        - `α` is calibrated per watershed using held-out data (optimises Brier Skill Score)
        - Default `α = 0.6` — temporal signal slightly dominates
        - Output classified into CWC-style alert levels: 🟢 Green → 🟡 Yellow → 🟠 Orange → 🔴 Red
        """)

        st.markdown("---")

        # Evaluation
        st.markdown("##### 📏 Evaluation Metrics")
        col_ev1, col_ev2 = st.columns(2)
        with col_ev1:
            st.markdown("""
            **Temporal (LSTM)**
            - **NSE** (Nash-Sutcliffe Efficiency)
            - **KGE** (Kling-Gupta Efficiency)
            - RMSE · MAE · PBIAS
            - Peak magnitude & timing error
            - Lead-time degradation analysis
            """)
        with col_ev2:
            st.markdown("""
            **Spatial (XGBoost)**
            - **AUC-ROC** · Brier Score
            - IoU (flood extent vs SAR mask)
            - POD (Prob. of Detection)
            - FAR (False Alarm Ratio)
            - CSI (Critical Success Index)
            """)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 2 — Data Sources
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_data:
        st.markdown("#### Data Ingestion & Feature Engineering")
        st.markdown("""
        ```
        ┌────────────────────────────────────────────────────────────────┐
        │                   DATA INGESTION  (src/ingestion/)             │
        │                                                                │
        │  glofas.py ─────────── GloFAS v4 river discharge (CDSE API)   │
        │  sentinel_sar.py ───── Sentinel-1 SAR  VV/VH (GEE Community) │
        │  dem_loader.py ─────── ALOS PALSAR 30 m DEM                   │
        └─────────────────────────────┬──────────────────────────────────┘
                                      │
        ┌─────────────────────────────▼──────────────────────────────────┐
        │              FEATURE ENGINEERING  (src/features/)              │
        │                                                                │
        │  terrain.py ──── TWI · slope · flow accumulation · curvature  │
        │  sar_processing.py ── SAR water masks · VV/VH ratio · NDWI   │
        │  precipitation.py ── Antecedent Precipitation Index (3-30 d)  │
        │  lulc_change.py ──── Land-use / land-cover delta via GEE     │
        └────────────────────────────────────────────────────────────────┘
        ```
        """)
        st.markdown("""
        **🔐 Auth:** Copernicus CDSE token-based OAuth2 (`src/utils/copernicus_auth.py`)

        **Geospatial tooling:** GDAL · Rasterio · GeoPandas · ECCODES
        """)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # TAB 3 — Deployment
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━
    with tab_deploy:
        st.markdown("#### Serving & Deployment")
        st.markdown("""
        ```
        ┌────────────────────────────────────────────────────────────────┐
        │       FastAPI  (api/)  — port 8000                             │
        │  POST /predict/water-level  · /susceptibility · /combined     │
        │  GET  /predict/metrics · /gauges/stations · /health           │
        │  WS   /ws/alerts (real-time push)                              │
        └──────────────────────────┬─────────────────────────────────────┘
                                   │  HTTP / WebSocket
        ┌──────────────────────────▼─────────────────────────────────────┐
        │       Streamlit Dashboard  (dashboard/)  — port 8501           │
        │  🌊 Live Monitor  ·  🗺️ Risk Maps  ·  📈 Forecasts  ·  📊 Analytics  │
        └──────────────────────────┬─────────────────────────────────────┘
                                   │
        ┌──────────────────────────▼─────────────────────────────────────┐
        │       Docker Compose  (docker/)                                │
        │  flood-api (CUDA GPU)  ·  flood-dashboard  ·  bridge network  │
        │  NVIDIA RTX 4050  ·  Shared volumes: /data /models /outputs   │
        └────────────────────────────────────────────────────────────────┘
        ```
        """)
