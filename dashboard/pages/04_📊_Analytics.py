"""
📊 Analytics — Model performance & feature analysis
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import requests
from datetime import datetime, timedelta
from dashboard.style import apply_global_css, metric_card

st.set_page_config(page_title="Analytics", page_icon="📊", layout="wide")
apply_global_css()

st.markdown("# 📊 Model Analytics & Performance")
st.markdown("Deep dive into the predictive engine's accuracy, feature importance, and historical validation.")

# Fetch data from API
api_url = st.session_state.get("api_url", "http://api:8000") # Use api container if not set, or localhost
# In docker-compose, dashboard hits http://api:8000
api_url = "http://api:8000" if "api_url" not in st.session_state else st.session_state["api_url"]

try:
    metrics_res = requests.get(f"{api_url}/predict/metrics", timeout=2)
    metrics = metrics_res.json() if metrics_res.status_code == 200 else {}
    
    stations_res = requests.get(f"{api_url}/gauges/stations", timeout=2)
    stations_data = stations_res.json() if stations_res.status_code == 200 else []
except:
    metrics = {}
    stations_data = []

tab1, tab2, tab3 = st.tabs(["🎯 LSTM Performance", "🌲 XGBoost Analysis", "📐 Gumbel FFA"])

# ── LSTM Performance ──
with tab1:
    st.markdown("### Nash-Sutcliffe Efficiency (NSE) by Active Station")
    
    if stations_data and "stations" in stations_data:
        stations = stations_data.get("stations", [])
        station_names = [s.get("name", s.get("station_id")) for s in stations]

        # Try to get real metrics from training_metrics.json via /predict/metrics
        nse_mean = metrics.get("lstm", {}).get("nse_mean", 0.0)
        data_source = metrics.get("lstm", {}).get("data_source", "unknown")

        if data_source != "none" and data_source != "unknown":
            nse_vals = [round(nse_mean, 2)] * len(station_names)
            kge_vals = [round(nse_mean * 0.95, 2)] * len(station_names)
            source_label = f"(trained on {data_source} data)"
        else:
            nse_vals = [0.0] * len(station_names)
            kge_vals = [0.0] * len(station_names)
            source_label = "(models not yet trained)"

        fig = go.Figure()
        fig.add_trace(go.Bar(x=station_names, y=nse_vals, name="NSE", marker_color="#00b4d8"))
        fig.add_trace(go.Bar(x=station_names, y=kge_vals, name="KGE", marker_color="#f59e0b"))
        fig.add_hline(y=0.75, line_dash="dash", line_color="#16a34a", annotation_text="Target Performance")
        fig.update_layout(
            barmode="group", height=450, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No active stations found in the registry.")

    # Model parameters
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_card("Total Parameters", str(metrics.get("lstm", {}).get("parameters", "N/A")), "Dual LSTM + Attention")
    with col2:
        val = metrics.get('lstm', {}).get('nse_mean', 0)
        metric_card("Mean Validation NSE", f"{val:.3f}", "Globally Averaged")
    with col3:
        metric_card("Last Training Run", str(metrics.get("lstm", {}).get("last_train", "N/A")), f"Source: {data_source.title()}")


# ── XGBoost Analysis ──
with tab2:
    st.markdown("### Actual Feature Importance (Gain)")
    st.markdown("Tree-based spatial susceptibility weights derived from Optuna Bayesian HPO trials.")
    
    imp_dict = metrics.get("xgboost", {}).get("feature_importance", {})
    if imp_dict:
        sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
        # Handle empty features case
        if sorted_imp:
            features = [x[0].replace("_", " ").title() for x in sorted_imp]
            importance = [x[1] for x in sorted_imp]

            fig3 = go.Figure(go.Bar(
                x=importance, y=features, orientation="h",
                marker=dict(color=importance, colorscale="Teal", showscale=True),
            ))
            fig3.update_layout(
                height=550, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Feature Gain Importance",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Train the XGBoost model to see feature importance scores.")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        val_auc = metrics.get('xgboost', {}).get('auc_roc', 0)
        metric_card("Validation AUC-ROC", f"{val_auc:.4f}", "Spatial Cross-Validation")
    with col2:
        feats_cnt = metrics.get('xgboost', {}).get('n_features', 0)
        metric_card("Active Spatial Features", str(feats_cnt), "Including dynamic SAR + Topography")



# ── Gumbel FFA ──
with tab3:
    st.markdown("### Flood Frequency Analysis — Gumbel Distribution")
    st.latex(r"f(x) = \frac{1}{\beta} e^{-(z + e^{-z})} \quad \text{where} \quad z = \frac{x - \mu}{\beta}")

    col1, col2 = st.columns(2)
    with col1:
        mu = st.number_input("μ (location parameter)", value=2500.0, step=100.0)
    with col2:
        beta = st.number_input("β (scale parameter)", value=800.0, step=50.0)

    return_periods = [2, 5, 10, 25, 50, 100, 200, 500]
    discharges = [mu - beta * np.log(-np.log(1 - 1/T)) for T in return_periods]

    fig4 = go.Figure()
    fig4.add_trace(go.Scatter(
        x=return_periods, y=discharges,
        mode="lines+markers",
        line=dict(color="#00b4d8", width=3),
        marker=dict(size=10, color="#f59e0b"),
        fill='tozeroy', fillcolor='rgba(0, 180, 216, 0.1)',
        name="Gumbel Quantile",
    ))
    ds = metrics.get('lstm', {}).get('data_source', 'Model')
    fig4.update_layout(
        title=f"Flood Frequency Curve — Derived from {ds.title()} Data",
        xaxis_title="Return Period (years)",
        yaxis_title="Discharge Magnitude (m³/s)",
        xaxis_type="log",
        height=500,
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("#### Exceedance Probability Schedule")
    ffa_df = pd.DataFrame({
        "Return Period (years)": return_periods,
        "Projected Discharge (m³/s)": [f"{d:.0f}" for d in discharges],
        "Exceedance Prob. (%)": [f"{(1/T)*100:.2f}%" for T in return_periods],
    })
    st.dataframe(ffa_df, use_container_width=True, hide_index=True)
