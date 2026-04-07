# 🌊 Flood Risk Prediction System — India

End-to-end flood forecasting and spatial risk mapping for the Indian subcontinent, powered by a hybrid **LSTM + XGBoost** architecture with real satellite imagery, terrain analysis, and India-specific hydrological data.

> **This system models physics, not just patterns.** Every module encodes the hydro-meteorological link: Rainfall → Infiltration → Runoff → Flood.

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       DATA INGESTION LAYER                       │
│  Sentinel-1 SAR  │  India-WRIS/CWC  │  IMD/GPM  │  SMAP/ERA5  │
│  (STAC API)      │  (Stream Gauges)  │ (Rainfall) │ (Soil Moist)│
└────────┬─────────┴────────┬──────────┴─────┬──────┴──────┬──────┘
         │                  │                │             │
┌────────▼──────────────────▼────────────────▼─────────────▼──────┐
│                    FEATURE ENGINEERING                           │
│  TWI │ Slope │ Flow Accum │ API │ SAR Water Masks │ LULC Change │
└─────────────┬──────────────────────────┬────────────────────────┘
              │                          │
┌─────────────▼────────┐   ┌─────────────▼────────────────────────┐
│   LSTM (PyTorch)      │   │    XGBoost (GPU-accelerated)         │
│   Hindcast-Forecast   │   │    Spatial Susceptibility Mapping    │
│   + Temporal Attention│   │    + Optuna HPO                      │
└──────────┬────────────┘   └──────────────┬──────────────────────┘
           │                               │
┌──────────▼───────────────────────────────▼──────────────────────┐
│              ENSEMBLE COMBINER (α-weighted fusion)               │
│         P_flood = α·P_temporal + (1-α)·P_spatial                │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│   FastAPI (REST + WebSocket)  →  Streamlit Dashboard (4 pages)  │
│   Docker Compose: api + dashboard services                      │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 India-Specific Data Sources

| Layer | Source | Resolution | Why |
|-------|--------|------------|-----|
| **Elevation** | ALOS PALSAR RTC | 12.5m | Sweet spot for Indian terrain — Himalayas to urban nallahs |
| **Rainfall** | IMD Gridded + NASA GPM | 0.25° / 0.1° | IMD for daily, GPM for 30-min real-time |
| **Water Level** | India-WRIS / CWC | 15-min telemetry | Ground truth for LSTM training and alerts |
| **Soil Moisture** | NASA SMAP | 9 km daily | Pre-saturation index — is the ground already soaked? |
| **SAR Imagery** | Sentinel-1 (STAC) | 10m | Sees through clouds during storms — detects standing water |
| **Land Cover** | ESA WorldCover | 10m | Urbanization = #1 driver of flash flood risk |
| **Reanalysis** | ERA5 (CDS) | 0.25° hourly | Multi-layer soil moisture, runoff, snowmelt |

---

## 🧮 Mathematical Foundations

### Gumbel Distribution (Flood Frequency Analysis)

```
f(x) = (1/β) × exp(-(z + exp(-z)))
z = (x - μ) / β

μ = x̄ - 0.5772·β     (location — Euler-Mascheroni)
β = (√6/π)·σ_x       (scale)

Return period discharge:  x_T = μ - β·ln(-ln(1 - 1/T))
```

### Topographic Wetness Index

```
TWI = ln(a / tan(β))
```

High TWI (>12) = flat + large upslope area = water accumulates = flood risk.

### Nash-Sutcliffe Efficiency

```
NSE = 1 - Σ(Qo - Qs)² / Σ(Qo - Q̄o)²
```

| NSE | Quality |
|-----|---------|
| 1.0 | Perfect |
| >0.75 | Very good |
| 0.36–0.75 | Satisfactory |
| <0.36 | Unsatisfactory |

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone <repo-url>
cd "Flood Risk Prediction System"

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your API keys (NASA Earthdata, CDS, etc.)
```

### 3. Download DEM for your AOI

```bash
# Greater Chennai example
python scripts/download_dem.py --bbox 80.0 12.8 80.4 13.2 --name chennai

# Brahmaputra Basin
python scripts/download_dem.py --bbox 89.5 25.5 96.0 28.0 --name brahmaputra
```

### 4. Run Feature Pipeline

```bash
python scripts/preprocess_pipeline.py --bbox 80.0 12.8 80.4 13.2 --name chennai
```

### 5. Launch API & Dashboard

```bash
# Terminal 1: API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Dashboard
streamlit run dashboard/app.py --server.port 8501
```

### 6. Docker Deployment

```bash
cd docker
docker-compose up --build
# API: http://localhost:8000/docs
# Dashboard: http://localhost:8501
```

---

## 📁 Project Structure

```
Flood Risk Prediction System/
├── config/
│   ├── settings.py              # Central config (India data sources, model params)
│   └── logging_config.py        # Loguru structured logging
├── src/
│   ├── ingestion/
│   │   ├── sentinel_sar.py      # Sentinel-1 SAR (STAC, no GEE)
│   │   ├── stream_gauges.py     # India-WRIS + CWC + Gumbel FFA
│   │   ├── atmospheric.py       # IMD rainfall + GPM + ERA5
│   │   └── dem_loader.py        # ALOS PALSAR 12.5m + SMAP soil moisture
│   ├── features/
│   │   ├── terrain.py           # TWI, slope, flow accumulation (WhiteboxTools)
│   │   ├── precipitation.py     # Antecedent Precipitation Index
│   │   ├── sar_processing.py    # SAR → water masks (Otsu + Lee filter)
│   │   └── lulc_change.py       # ESA WorldCover change detection
│   ├── models/
│   │   ├── lstm_forecaster.py   # PyTorch LSTM (hindcast-forecast + attention)
│   │   ├── spatial_susceptibility.py  # XGBoost flood susceptibility
│   │   ├── ensemble.py          # α-weighted temporal+spatial fusion
│   │   └── training/
│   │       ├── train_lstm.py    # Walk-forward CV + Optuna HPO
│   │       └── train_xgboost.py # Spatial CV + raster-to-tabular
│   ├── evaluation/
│   │   └── metrics.py           # NSE, KGE, RMSE, IoU, FAR, POD, CSI
│   ├── geospatial/
│   │   ├── dem_processing.py    # Reproject, void fill, hillshade
│   │   ├── raster_utils.py      # Tiling, zonal stats, GeoTIFF I/O
│   │   └── vector_utils.py      # AOI loading, catchment attributes
│   └── utils/
│       ├── data_validation.py   # Pydantic schemas
│       └── scalability.py       # Dask distributed processing
├── api/
│   ├── main.py                  # FastAPI app (REST + WebSocket)
│   ├── schemas.py               # Request/response models
│   ├── dependencies.py          # Model caching + DI
│   └── routers/
│       ├── predictions.py       # /predict/* endpoints
│       ├── gauges.py            # /gauges/* endpoints
│       └── risk_maps.py         # /risk-map/* endpoints
├── dashboard/
│   ├── app.py                   # Streamlit main page
│   └── pages/
│       ├── 01_🌊_Live_Monitor.py
│       ├── 02_🗺️_Risk_Maps.py
│       ├── 03_📈_Forecasts.py
│       └── 04_📊_Analytics.py
├── docker/
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── docker-compose.yml
├── scripts/
│   ├── download_dem.py
│   └── preprocess_pipeline.py
├── tests/
│   ├── test_terrain.py
│   ├── test_lstm.py
│   ├── test_metrics.py
│   └── test_api.py
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## 🔬 Model Details

### LSTM Water Level Forecaster

- **Architecture**: Hindcast-Forecast dual-LSTM with multi-head temporal attention
- **Input**: 7-day lookback (168 hourly steps) of dynamic features + static catchment attributes
- **Output**: 72-hour probabilistic forecast (mean + σ for uncertainty)
- **Loss**: Gaussian Negative Log-Likelihood (learns both prediction and uncertainty)
- **Training**: Walk-forward CV, mixed-precision (FP16), gradient clipping, cosine LR

### XGBoost Spatial Susceptibility

- **Features**: 15 terrain + LULC + weather features per grid cell
- **Target**: Binary flood/no-flood from SAR-derived flood masks
- **HPO**: Optuna Bayesian search (50 trials)
- **Validation**: Spatial k-fold (leave-one-watershed-out) — no spatial leakage
- **GPU**: `tree_method="gpu_hist"` on RTX 4050

### Ensemble

```
P_flood(x,y,t) = α·P_temporal(t) + (1-α)·P_spatial(x,y)
```

α calibrated per-basin using held-out data (Brier Skill Score optimization).

---

## 🧪 Testing

```bash
# All tests
pytest tests/ -v

# Specific test suites
pytest tests/test_metrics.py -v    # Hydrological metrics
pytest tests/test_lstm.py -v       # LSTM model
pytest tests/test_terrain.py -v    # Terrain features
pytest tests/test_api.py -v        # API endpoints
```

---

## 📡 API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/predict/water-level` | LSTM forecast for a gauge station |
| `POST` | `/predict/susceptibility` | XGBoost flood probability for bbox |
| `POST` | `/predict/combined` | Ensemble prediction |
| `GET` | `/gauges/stations` | List India-WRIS stations |
| `GET` | `/gauges/live/{id}` | Real-time gauge reading |
| `GET` | `/gauges/historical/{id}` | Historical time series |
| `GET` | `/risk-map/{region}` | Pre-computed risk GeoTIFF |
| `WS` | `/ws/alerts` | Real-time flood alert push |
| `GET` | `/health` | Service health + model status |

Interactive docs: `http://localhost:8000/docs`

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| DL Framework | PyTorch (CUDA / RTX 4050) |
| Gradient Boosting | XGBoost (GPU-accelerated) |
| HPO | Optuna (Bayesian) |
| Geospatial | Rasterio, GDAL, GeoPandas, WhiteboxTools |
| Satellite Access | STAC API, stackstac, Planetary Computer |
| API | FastAPI + Uvicorn |
| Dashboard | Streamlit + Folium + Plotly |
| Deployment | Docker + Docker Compose |
| Scalability | Dask, rioxarray, Zarr |
| Logging | Loguru (structured JSON) |
| Validation | Pydantic v2 |

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
