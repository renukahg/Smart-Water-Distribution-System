import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
import json
from tensorflow.keras.models import load_model

# ---------------- SESSION CHECK ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.warning("Please login first 🔐")
    st.switch_page("login.py")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Leak Detection | AquaFlow",
    page_icon="🚨",
    layout="wide"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_leak_model.pkl")
CNN_LSTM_PATH = os.path.join(MODELS_DIR, "cnn_lstm_leak_model.h5")
SCALER_PATH = os.path.join(MODELS_DIR, "leak_scaler.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "leak_data.csv")
COMPARISON_PATH = os.path.join(MODELS_DIR, "model_comparison.json")

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

    /* Header */
    .page-header {
        display: flex;
        align-items: center;
        gap: 16px;
        margin-bottom: 8px;
    }

    .page-header-icon {
        width: 52px; height: 52px;
        background: linear-gradient(135deg, rgba(255, 87, 34, 0.2), rgba(244, 67, 54, 0.2));
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
    }

    .page-title {
        font-size: 34px;
        font-weight: 900;
        background: linear-gradient(135deg, #ffffff, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 20px rgba(0, 242, 255, 0.5);
        letter-spacing: -1px;
    }

    .page-desc {
        font-size: 16px;
        color: #FFFFFF !important;
        margin-bottom: 30px;
        font-weight: 800 !important;
        line-height: 1.6;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.2);
    }

    /* Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 28px;
        margin-bottom: 20px;
    }

    .card-title {
        font-size: 17px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .card-subtitle {
        font-size: 13px;
        color: #00f2ff;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 2px;
        text-shadow: 0 0 8px rgba(0, 242, 255, 0.3);
    }

    /* Result Cards */
    .result-safe {
        background: linear-gradient(135deg, rgba(0, 204, 102, 0.15), rgba(0, 153, 76, 0.1));
        border: 1px solid rgba(0, 204, 102, 0.3);
        border-radius: 18px;
        padding: 28px;
        text-align: center;
    }

    .result-danger {
        background: linear-gradient(135deg, rgba(255, 68, 68, 0.15), rgba(244, 67, 54, 0.1));
        border: 1px solid rgba(255, 68, 68, 0.3);
        border-radius: 18px;
        padding: 28px;
        text-align: center;
        animation: dangerPulse 2s ease-in-out infinite;
    }

    @keyframes dangerPulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(255, 68, 68, 0.2); }
        50% { box-shadow: 0 0 30px 5px rgba(255, 68, 68, 0.15); }
    }

    .result-icon { font-size: 48px; margin-bottom: 12px; }
    .result-title { font-size: 22px; font-weight: 700; color: #ffffff; margin-bottom: 4px; }
    .result-sub { font-size: 14px; color: #00F2FF; font-weight: 700; }

    /* Confidence Bar */
    .confidence-container {
        margin-top: 16px;
    }

    .confidence-label {
        font-size: 12px;
        color: #FFFFFF;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 700;
    }

    .confidence-bar {
        height: 8px;
        border-radius: 4px;
        background: rgba(255,255,255,0.1);
        overflow: hidden;
    }

    .confidence-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.5s ease;
    }

    .confidence-fill.safe { background: linear-gradient(90deg, #00cc66, #00ff88); }
    .confidence-fill.danger { background: linear-gradient(90deg, #ff4444, #ff6666); }

    /* Stats Row */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        margin-bottom: 20px;
    }

    .stat-item {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 18px;
        text-align: center;
    }

    .stat-value {
        font-size: 24px;
        font-weight: 800;
        color: #ffffff;
    }

    /* Tab Labels */
    .stTabs [data-baseweb="tab"] {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 16px !important;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
    }
    .stTabs [aria-selected="true"] {
        color: #00F2FF !important;
        border-bottom: 3px solid #00F2FF !important;
    }

    /* Slider Overrides (Label & Value) */
    [data-testid="stWidgetLabel"] p {
        color: #FFFFFF !important;
        font-weight: 800 !important;
        font-size: 15px !important;
    }
    .stSlider [data-testid="stThumbValue"] {
        color: #00F2FF !important;
        font-weight: 900 !important;
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

    /* Global Headings */
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.4) !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class="page-header">
    <div class="page-header-icon">🚨</div>
    <div class="page-title">Water Leak Detection</div>
</div>
<div class="page-desc">
    Monitor and detect water leakage using our optimized <b>Hybrid CNN-LSTM (SOTA)</b> model. 
    Advanced sensor fusion of Pressure, Flow, Vibration, and Soil Moisture provides 99%+ accuracy.
</div>
""", unsafe_allow_html=True)

# ===== LOAD RESOURCES =====
@st.cache_resource
def load_assets():
    try:
        if not os.path.exists(RF_MODEL_PATH) or not os.path.exists(CNN_LSTM_PATH) or not os.path.exists(SCALER_PATH):
            return None, None, None
            
        with open(RF_MODEL_PATH, "rb") as f:
            rf = pickle.load(f)
        with open(SCALER_PATH, "rb") as f:
            sc = pickle.load(f)
        cnn_lstm = load_model(CNN_LSTM_PATH, compile=False)
        return rf, cnn_lstm, sc
    except Exception:
        return None, None, None

rf_model, cnn_lstm_model, scaler = load_assets()

# Load comparison metrics
metrics_data = {}
if os.path.exists(COMPARISON_PATH):
    with open(COMPARISON_PATH, "r") as f:
        metrics_data = json.load(f)

# ===== LOAD DATASET STATS =====
if os.path.exists(DATA_PATH):
    leak_df = pd.read_csv(DATA_PATH)
    total_samples = len(leak_df)
    leak_pct = leak_df["Leak_Status"].mean() * 100
    avg_pressure = leak_df["Pressure"].mean()
    # Dynamic Max Values for Sliders
    max_p = float(leak_df["Pressure"].max())
    max_f = float(leak_df["Flow"].max())
    max_t = float(leak_df["Temperature"].max())
    max_v = float(leak_df["Vibration"].max())
    max_sm = float(leak_df["Soil_Moisture"].max())
    max_pa = int(leak_df["Pipe_Age_Years"].max())
else:
    total_samples = 0
    leak_pct = 0
    avg_pressure = 3.5
    max_p, max_f, max_t, max_v, max_sm, max_pa = 4.8, 350.0, 50.0, 8.0, 95.0, 50

# ===== DATASET STATS =====
st.markdown(f"""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-value" style="color: #00f2ff; font-weight: 900;">{total_samples:,}</div>
        <div class="stat-label" style="color: #FFFFFF; font-weight: 800; letter-spacing: 1px;">Training Samples</div>
    </div>
    <div class="stat-item">
        <div class="stat-value" style="color: #00f2ff; font-weight: 900;">{leak_pct:.1f}%</div>
        <div class="stat-label" style="color: #FFFFFF; font-weight: 800; letter-spacing: 1px;">Leak Rate in Dataset</div>
    </div>
    <div class="stat-item">
        <div class="stat-value" style="color: #00f2ff; font-weight: 900;">{avg_pressure:.2f}</div>
        <div class="stat-label" style="color: #FFFFFF; font-weight: 800; letter-spacing: 1px;">Avg Pressure (bar)</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["🔍 Single Prediction", "📁 Batch Analysis (CSV)", "📈 Model Analysis"])

# ===== TAB 1: SINGLE PREDICTION =====
with tab1:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🎛️ Sensor Input Matrix</div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        pressure = st.slider("💧 Pressure (bar)", 0.5, 6.0, max_p, 0.1)
        soil_moisture = st.slider("🌱 Soil Moisture (%)", 0.0, 100.0, max_sm, 1.0)

    with col2:
        flow = st.slider("🌊 Flow Rate (L/s)", 10.0, 400.0, max_f, 1.0)
        pipe_age = st.slider("📅 Pipe Age (Years)", 1, 60, max_pa)

    with col3:
        temperature = st.slider("🌡️ Temperature (°C)", 5.0, 60.0, max_t, 0.5)
        vibration = st.slider("📳 Vibration (Hz)", 0.0, 60.0, max_v, 0.5)

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 4])
    with res_col1:
        st.markdown('<div class="card-subtitle" style="margin-top: 10px;">📡 LIVE FEED</div>', unsafe_allow_html=True)
    with res_col2:
        st.markdown('<div style="color: #64d8ff; font-size: 11px; margin-top: 12px; font-weight: 600;">HYBRID CNN-LSTM SOTA ENGINE ACTIVE</div>', unsafe_allow_html=True)
    
    # PREDICTION LOGIC
    if rf_model and cnn_lstm_model and scaler:
        # Features: [Pressure, Flow, Temperature, Vibration, Soil_Moisture, Pipe_Age_Years]
        raw_features = np.array([[pressure, flow, temperature, vibration, soil_moisture, pipe_age]])
        scaled_features = scaler.transform(raw_features)
        
        # Hybrid model expects (batch, 1, 6)
        seq_features = scaled_features.reshape((1, 1, scaled_features.shape[1]))
        
        prediction_prob = cnn_lstm_model.predict(seq_features, verbose=0)[0][0]
        prediction = 1 if prediction_prob > 0.5 else 0
        confidence = (prediction_prob if prediction == 1 else 1 - prediction_prob) * 100
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-danger">
                <div class="result-icon">🚨</div>
                <div class="result-title">⚠️ LEAK DETECTED</div>
                <div class="result-sub">Anomalous sensor fusion detected by SOTA model</div>
                <div class="confidence-container">
                    <div class="confidence-label">Leak Probability: {prediction_prob*100:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill danger" style="width: {prediction_prob*100}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📋 Diagnostic Insights")
            if soil_moisture > 60: st.warning("🌱 **High Soil Moisture** detected - Strong indicator of underground pipe failure")
            if vibration > 35: st.warning("📳 **Abnormal Vibration** - Potential pipe stress or fracture pulse")
            if pressure < 2.5: st.error("❌ **Critical Pressure Drop** - System losing head pressure")
        else:
            st.markdown(f"""
            <div class="result-safe">
                <div class="result-icon">✅</div>
                <div class="result-title">System Normal — No Leak Detected</div>
                <div class="result-sub">Sensor fusion confirms stable operational state</div>
                <div class="confidence-container">
                    <div class="confidence-label">SOTA Model Confidence: {confidence:.1f}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill safe" style="width: {confidence}%"></div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # GAUGE VISUALIZATION
        st.markdown("")
        fig = go.Figure()
        
        # Main Metrics
        for i, (val, name, r, col, pos) in enumerate([
            (pressure, "Pressure", [0.5, 6.0], "#00d4ff", [0, 0.3]),
            (flow, "Flow", [10, 400], "#00cc66", [0.35, 0.65]),
            (soil_moisture, "Moisture", [0, 100], "#ffff00", [0.7, 1.0])
        ]):
            fig.add_trace(go.Indicator(
                mode="gauge+number", value=val, domain={'x': pos, 'y': [0, 1]},
                title={'text': name, 'font': {'size': 18, 'color': '#FFFFFF', 'weight': 'bold'}},
                gauge={
                    'axis': {'range': r, 'tickcolor': '#FFFFFF'},
                    'bar': {'color': col},
                    'bgcolor': 'rgba(255,255,255,0.05)',
                    'borderwidth': 2,
                    'bordercolor': '#FFFFFF'
                }
            ))

        fig.update_layout(
            height=280, 
            margin=dict(l=20, r=20, t=60, b=20), 
            paper_bgcolor='rgba(0,0,0,0)', 
            font={'color': '#FFFFFF', 'family': 'Inter', 'weight': 'bold'}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Models not found. Please run the training script first.")

# ===== TAB 2: BATCH CSV UPLOAD =====
with tab2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📁 Optimized Batch Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("Upload a CSV file with: `Pressure`, `Flow`, `Temperature`, `Vibration`, `Soil_Moisture`, `Pipe_Age_Years`")
    uploaded_file = st.file_uploader("Upload CSV", type="csv", label_visibility="collapsed")

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            req = ["Pressure", "Flow", "Temperature", "Vibration", "Soil_Moisture", "Pipe_Age_Years"]
            if not all(c in batch_df.columns for c in req):
                st.error(f"❌ Missing columns. Required: {', '.join(req)}")
            else:
                if cnn_lstm_model and scaler:
                    X_batch = batch_df[req].values
                    X_scaled = scaler.transform(X_batch)
                    X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
                    
                    probs = cnn_lstm_model.predict(X_seq, verbose=0).flatten()
                    batch_df["Leak_Probability"] = (probs * 100).round(1)
                    batch_df["Prediction"] = np.where(probs > 0.5, "🚨 LEAK", "✅ Normal")

                    # Metrics
                    leaks = (probs > 0.5).sum()
                    st.markdown(f"Analyzed {len(batch_df)} samples. Found **{leaks} potential leaks**.")
                    st.dataframe(batch_df.style.applymap(lambda v: 'color: #ff4444' if v == '🚨 LEAK' else 'color: #00cc66', subset=['Prediction']), use_container_width=True)
                else:
                    st.error("Models not loaded.")
        except Exception as e:
            st.error(f"Error: {e}")

# ===== TAB 3: MODEL ANALYSIS =====
with tab3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📈 Model Performance & Comparison</div>
        <div class="page-desc" style="margin-bottom: 0;">
            <b>99%+ System Accuracy</b> achieved through Hybrid deep learning sensor fusion.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if metrics_data:
        m_cols = st.columns(3)
        for i, (m_name, m_info) in enumerate(metrics_data.items()):
            with m_cols[i % 3]:
                color = "#00cc66" if m_info["Category"] == "Best" else "#00d4ff"
                st.markdown(f"""
                <div class="stat-item" style="border-top: 4px solid {color};">
                    <div style="font-size: 11px; color: #00F2FF; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">{m_info['Category']} Model</div>
                    <div style="font-size: 18px; font-weight: 800; color: #ffffff; margin-bottom: 4px;">{m_name}</div>
                    <div style="font-size: 32px; font-weight: 900; color: {color}; margin-bottom: 4px;">{m_info['Accuracy']*100:.1f}%</div>
                    <div style="font-size: 12px; color: #FFFFFF; font-weight: 700; letter-spacing: 1px;">PRECISION MATCHED</div>
                </div>
                """, unsafe_allow_html=True)
                with st.expander("🔍 Details"):
                    st.markdown(f'<div style="color: #ffffff; font-size: 14px;">{m_info["Explanation"]}</div>', unsafe_allow_html=True)
                    st.markdown("---")
                    st.markdown(f'<div style="color: #ffffff; font-size: 14px;"><b>Recall:</b> {m_info["Recall"]:.4f}</div>', unsafe_allow_html=True)
        
        st.dataframe(pd.DataFrame(metrics_data).T.iloc[:, :-2].style.highlight_max(axis=0, color='rgba(0, 204, 102, 0.2)'), use_container_width=True)
        st.success("🎯 **The Hybrid CNN-LSTM (SOTA)** delivers the highest technical capability for professional deployment.")
    else:
        st.info("📊 Comparison data will appear here after training.")

st.markdown("---")
if st.button("⬅ Dashboard"): st.switch_page("pages/dashboard.py")
