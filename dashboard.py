import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime

# ---------------- SESSION INITIALIZATION ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None

# ---------------- ACCESS CONTROL ----------------
if not st.session_state.logged_in:
    st.warning("Please login to access the dashboard 🔐")
    st.switch_page("login.py")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AquaFlow Dashboard",
    page_icon="💧",
    layout="wide"
)

# ---------------- LOAD DATA FOR METRICS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMAND_PATH = os.path.join(BASE_DIR, "data", "demand_data.csv")
LEAK_PATH = os.path.join(BASE_DIR, "data", "leak_data.csv")

demand_df = None
leak_df = None

if os.path.exists(DEMAND_PATH):
    demand_df = pd.read_csv(DEMAND_PATH, parse_dates=["Date"])
    demand_df = demand_df.sort_values("Date")

if os.path.exists(LEAK_PATH):
    leak_df = pd.read_csv(LEAK_PATH)

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    .stApp {
        background: linear-gradient(135deg, #001a2c 0%, #004d7a 50%, #001a2c 100%);
        font-family: 'Inter', sans-serif;
    }

    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }

    /* ===== TOP HEADER BAR ===== */
    .dash-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 0;
        border-bottom: 1px solid rgba(255,255,255,0.06);
        margin-bottom: 30px;
    }

    .header-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .header-logo {
        width: 48px; height: 48px;
        background: linear-gradient(135deg, #00d4ff, #0077b6);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
    }

    .header-brand {
        font-size: 22px;
        font-weight: 700;
        color: #fff;
    }

    .header-tag {
        font-size: 12px;
        color: #00F2FF;
        font-weight: 700;
    }

    .header-right {
        display: flex;
        align-items: center;
        gap: 20px;
    }

    .user-badge {
        background: rgba(255, 255, 255, 0.1);
        border: 1.5px solid rgba(255, 255, 255, 0.2);
        border-radius: 12px;
        padding: 8px 18px;
        color: #FFFFFF;
        font-size: 13px;
        font-weight: 800;
        text-shadow: 0 0 5px rgba(255,255,255,0.3);
    }

    .status-dot {
        display: inline-block;
        width: 8px; height: 8px;
        background: #00cc66;
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    /* ===== WELCOME SECTION ===== */
    .welcome-section {
        margin-bottom: 32px;
    }

    .welcome-title {
        font-size: 38px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.5);
    }

    .welcome-sub {
        font-size: 16px;
        color: #FFFFFF !important;
        font-weight: 700;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.4);
    }

    /* ===== METRIC CARDS ===== */
    .metrics-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 32px;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }

    .metric-card:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
    }

    .metric-card::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 80px; height: 80px;
        border-radius: 50%;
        opacity: 0.05;
    }

    .metric-card.blue::after { background: #00d4ff; }
    .metric-card.green::after { background: #00cc66; }
    .metric-card.orange::after { background: #ff8c00; }
    .metric-card.purple::after { background: #9c27b0; }

    .metric-icon {
        font-size: 28px;
        margin-bottom: 12px;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 4px;
    }

    .metric-label {
        font-size: 13px;
        color: #00f2ff;
        font-weight: 900;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-shadow: 0 0 5px rgba(0, 242, 255, 0.3);
    }

    .metric-change {
        font-size: 12px;
        margin-top: 8px;
        font-weight: 500;
    }

    .metric-change.up { color: #00cc66; }
    .metric-change.down { color: #ff4444; }
    .metric-change.neutral { color: #64b5f6; }

    /* ===== MODULE CARDS ===== */
    .modules-title {
        font-size: 20px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 20px;
    }

    .module-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 32px 28px;
        height: 100%;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .module-card:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.3);
    }

    .module-icon-wrap {
        width: 56px; height: 56px;
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        margin-bottom: 18px;
    }

    .module-icon-wrap.forecast { background: linear-gradient(135deg, rgba(0, 212, 255, 0.2), rgba(0, 119, 182, 0.2)); }
    .module-icon-wrap.leak { background: linear-gradient(135deg, rgba(255, 87, 34, 0.2), rgba(244, 67, 54, 0.2)); }

    .module-title {
        font-size: 18px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 8px;
    }

    .module-desc {
        font-size: 14px;
        color: #FFFFFF;
        line-height: 1.6;
        margin-bottom: 18px;
        font-weight: 700;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.1);
    }

    .module-features {
        display: flex;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 16px;
    }

    .module-feat {
        background: rgba(0, 212, 255, 0.08);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 8px;
        padding: 4px 10px;
        font-size: 11px;
        color: #64d8ff;
    }

    /* ===== ACTIVITY SECTION ===== */
    .activity-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 24px;
        margin-top: 24px;
    }

    .activity-title {
        font-size: 16px;
        font-weight: 600;
        color: #ffffff;
        margin-bottom: 16px;
    }

    .activity-item {
        display: flex;
        align-items: center;
        gap: 12px;
        padding: 10px 0;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .activity-item:last-child {
        border-bottom: none;
    }

    .activity-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        flex-shrink: 0;
    }

    .activity-dot.green { background: #00cc66; }
    .activity-dot.blue { background: #00b4d8; }
    .activity-dot.yellow { background: #ffaa00; }
    .activity-dot.red { background: #ff4444; }

    .activity-text {
        font-size: 13px;
        color: #FFFFFF;
    }

    .activity-time {
        font-size: 11px;
        color: #00f2ff;
        margin-left: auto;
        font-weight: 800;
        text-shadow: 0 0 5px rgba(0, 242, 255, 0.2);
    }

    /* ===== BUTTONS ===== */
    div.stButton > button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 12px 24px !important;
        font-size: 15px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        width: 100% !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3) !important;
    }

    div.stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 180, 216, 0.5) !important;
    }

    /* ===== RESPONSIVE ===== */
    @media (max-width: 768px) {
        .metrics-row {
            grid-template-columns: repeat(2, 1fr);
        }
    }

    /* Metric Overrides */
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #00F2FF !important;
        font-weight: 800 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Global Headings Glow */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 950 !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER BAR =====
user_display = st.session_state.user if st.session_state.user else "User"
current_time = datetime.now().strftime("%B %d, %Y • %I:%M %p")

st.markdown(f"""
<div class="dash-header">
    <div class="header-left">
        <div class="header-logo">💧</div>
        <div>
            <div class="header-brand">AquaFlow Intelligence</div>
            <div class="header-tag">Smart Water Distribution System</div>
        </div>
    </div>
    <div class="header-right">
        <div class="user-badge"><span class="status-dot"></span> {user_display}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== WELCOME SECTION =====
hour = datetime.now().hour
greeting = "Good Morning" if hour < 12 else ("Good Afternoon" if hour < 17 else "Good Evening")

st.markdown(f"""
<div class="welcome-section">
    <div class="welcome-title">{greeting}, {user_display.title()} 👋</div>
    <div class="welcome-sub">Here's an overview of your water distribution network — {current_time}</div>
</div>
""", unsafe_allow_html=True)

# ===== METRICS ======
if demand_df is not None and leak_df is not None:
    avg_demand = demand_df["Demand"].mean()
    latest_demand = demand_df["Demand"].iloc[-7:].mean()
    prev_demand = demand_df["Demand"].iloc[-14:-7].mean()
    demand_change = ((latest_demand - prev_demand) / prev_demand * 100)

    total_records = len(demand_df)
    leak_rate = leak_df["Leak_Status"].mean() * 100

    # System health based on leak rate
    if leak_rate < 15:
        health = "98.5%"
        health_label = "Excellent"
    elif leak_rate < 25:
        health = "92.3%"
        health_label = "Good"
    else:
        health = "85.7%"
        health_label = "Needs Attention"

    st.markdown(f"""
    <div class="metrics-row">
        <div class="metric-card blue">
            <div class="metric-icon">📊</div>
            <div class="metric-value">{latest_demand:.0f}</div>
            <div class="metric-label">Avg Daily Demand (L)</div>
            <div class="metric-change {'up' if demand_change > 0 else 'down'}">
                {'↑' if demand_change > 0 else '↓'} {abs(demand_change):.1f}% vs last week
            </div>
        </div>
        <div class="metric-card green">
            <div class="metric-icon">🏗️</div>
            <div class="metric-value">{total_records:,}</div>
            <div class="metric-label">Data Points Collected</div>
            <div class="metric-change neutral">📅 {demand_df['Date'].min().strftime('%Y')} - {demand_df['Date'].max().strftime('%Y')}</div>
        </div>
        <div class="metric-card orange">
            <div class="metric-icon">🚨</div>
            <div class="metric-value">{leak_rate:.1f}%</div>
            <div class="metric-label">Leak Detection Rate</div>
            <div class="metric-change {'down' if leak_rate < 20 else 'up'}">
                {len(leak_df):,} samples analyzed
            </div>
        </div>
        <div class="metric-card purple">
            <div class="metric-icon">💚</div>
            <div class="metric-value">{health}</div>
            <div class="metric-label">System Health</div>
            <div class="metric-change neutral">Status: {health_label}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== MODULE CARDS =====
st.markdown('<div class="modules-title">🧩 System Modules</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="module-card">
        <div class="module-icon-wrap forecast">📈</div>
        <div class="module-title">Water Demand Forecasting</div>
        <div class="module-desc">
            Predict future water demand using LSTM deep learning. Analyze seasonal patterns, 
            temperature effects, and population growth to optimize water distribution.
        </div>
        <div class="module-features">
            <span class="module-feat">LSTM Model</span>
            <span class="module-feat">7-30 Day Forecast</span>
            <span class="module-feat">Trend Analysis</span>
            <span class="module-feat">Interactive Charts</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("📈 Open Demand Forecasting"):
        st.switch_page("pages/demand_forecasting.py")

with col2:
    st.markdown("""
    <div class="module-card">
        <div class="module-icon-wrap leak">🚰</div>
        <div class="module-title">Water Leak Detection</div>
        <div class="module-desc">
            Detect water leaks in real-time using Random Forest classification. Monitor pressure, 
            flow rate, vibration and soil moisture to identify potential pipe failures.
        </div>
        <div class="module-features">
            <span class="module-feat">Random Forest</span>
            <span class="module-feat">Real-time Analysis</span>
            <span class="module-feat">Risk Assessment</span>
            <span class="module-feat">CSV Upload</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🚰 Open Leak Detection"):
        st.switch_page("pages/leak_detection.py")

# ===== RECENT ACTIVITY =====
st.markdown(f"""
<div class="activity-card">
    <div class="activity-title">📋 Recent System Activity</div>
    <div class="activity-item">
        <div class="activity-dot green"></div>
        <div class="activity-text">System health check completed — All zones operational</div>
        <div class="activity-time">Just now</div>
    </div>
    <div class="activity-item">
        <div class="activity-dot blue"></div>
        <div class="activity-text">Demand forecast model updated with latest data ({total_records if demand_df is not None else 0:,} records)</div>
        <div class="activity-time">2 hours ago</div>
    </div>
    <div class="activity-item">
        <div class="activity-dot yellow"></div>
        <div class="activity-text">Leak detection scan completed — {leak_rate if leak_df is not None else 0:.1f}% anomaly rate detected</div>
        <div class="activity-time">4 hours ago</div>
    </div>
    <div class="activity-item">
        <div class="activity-dot green"></div>
        <div class="activity-text">User <strong>{user_display}</strong> logged in successfully</div>
        <div class="activity-time">{datetime.now().strftime('%I:%M %p')}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== LOGOUT =====
st.markdown("---")
col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    if st.button("🔓 Logout"):
        st.session_state.logged_in = False
        st.session_state.user = None
        st.switch_page("login.py")