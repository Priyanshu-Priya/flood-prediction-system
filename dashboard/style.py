import streamlit as st

def apply_global_css():
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Dark theme enhancements */
    .stApp {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1a3e 50%, #0d1117 100%);
        color: #e2e8f0;
    }

    /* Typography */
    h1, h2, h3, h4, h5 {
        background: linear-gradient(90deg, #00b4d8, #48cae4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }

    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        transition: transform 0.3s ease, box-shadow 0.3s ease, border-color 0.3s;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        margin-bottom: 1rem;
    }

    .glass-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px 0 rgba(0, 180, 216, 0.2);
        border-color: rgba(0, 180, 216, 0.5);
    }
    
    .glass-metric-label {
        font-size: 0.9rem;
        color: #94a3b8;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .glass-metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #caf0f8, #00b4d8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-top: 0.2rem;
    }

    .glass-metric-subtext {
        font-size: 0.8rem;
        color: #64748b;
        margin-top: 0.2rem;
    }

    /* Compact Metric Variants */
    .compact-card {
        padding: 0.8rem 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    .compact-card .glass-metric-label {
        font-size: 0.75rem;
    }
    .compact-card .glass-metric-value {
        font-size: 1.5rem;
        margin-top: 0;
    }
    .compact-card .glass-metric-subtext {
        font-size: 0.7rem;
    }

    /* Streamlit Native Overrides */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0 0;
        gap: 10px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: rgba(0, 180, 216, 0.1);
        border-bottom-color: #00b4d8 !important;
        color: #00b4d8 !important;
    }
    
    .stButton>button {
        background: linear-gradient(90deg, #0077b6, #00b4d8);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.5rem;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.4);
        color: white;
    }

    /* Sidebar mapping */
    [data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.8) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
</style>
    """, unsafe_allow_html=True)

def metric_card(label: str, value: str, subtext: str = "", compact: bool = False):
    card_class = "glass-card compact-card" if compact else "glass-card"
    st.markdown(f"""
    <div class="{card_class}">
        <div class="glass-metric-label">{label}</div>
        <div class="glass-metric-value">{value}</div>
        <div class="glass-metric-subtext">{subtext}</div>
    </div>
    """, unsafe_allow_html=True)
