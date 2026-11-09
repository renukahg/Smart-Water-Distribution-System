import streamlit as st
import numpy as np
import pandas as pd
import os
import pickle
import plotly.graph_objects as go
import plotly.express as px
from tensorflow.keras.models import load_model

# ---------------- SESSION CHECK ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.warning("Please login first 🔐")
    st.switch_page("login.py")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Demand Forecasting | AquaFlow",
    page_icon="📈",
    layout="wide"
)

# ---------------- PATHS ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "demand_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "demand_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

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
        background: linear-gradient(135deg, rgba(0, 212, 255, 0.3), rgba(0, 119, 182, 0.3));
        border-radius: 16px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        border: 1px solid rgba(0, 212, 255, 0.4);
    }

    .page-title {
        font-size: 34px;
        font-weight: 950;
        background: linear-gradient(135deg, #ffffff, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 0 25px rgba(0, 242, 255, 0.7);
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

    /* Stats Row */
    .stats-row {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 14px;
        margin-bottom: 28px;
    }

    .stat-item {
        background: rgba(255, 255, 255, 0.04);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        text-align: center;
        transition: all 0.3s ease;
    }

    .stat-item:hover {
        border-color: rgba(0, 212, 255, 0.3);
        transform: translateY(-2px);
    }

    .stat-icon { font-size: 24px; margin-bottom: 8px; }

    .stat-value {
        font-size: 26px;
        font-weight: 800;
        color: #ffffff;
    }

    .stat-label {
        font-size: 11px;
        color: #FFFFFF;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
        font-weight: 900;
    }

    .stat-sub {
        font-size: 11px;
        color: #FFFFFF;
        margin-top: 4px;
        font-weight: 700;
        text-shadow: 0 0 5px rgba(255, 255, 255, 0.2);
    }

    /* Glass Card */
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
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 24px !important;
        font-size: 16px !important;
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

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255,255,255,0.03);
        border-radius: 14px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #FFFFFF;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
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

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Alerts & Success Messages */
    .stAlert p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
        text-shadow: 0 0 5px rgba(255,255,255,0.3);
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

    /* Expander Styling */
    .stExpander {
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        background: rgba(255, 255, 255, 0.03) !important;
        border-radius: 12px !important;
    }
    .stExpander summary p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ===== HEADER =====
st.markdown("""
<div class="page-header">
    <div class="page-header-icon">📈</div>
    <div class="page-title">Water Demand Forecasting</div>
</div>
<div class="page-desc">
    Predict future water demand using LSTM deep learning. Analyze historical patterns, 
    seasonal trends, and environmental factors for proactive water management.
</div>
""", unsafe_allow_html=True)

# ===== FILE CHECKS =====
if not os.path.exists(DATA_PATH):
    st.error("❌ demand_data.csv not found in data/ folder")
    st.stop()

if not os.path.exists(MODEL_PATH):
    st.error("❌ Model not found. Please train the model first.")
    st.stop()

if not os.path.exists(SCALER_PATH):
    st.error("❌ Scaler file not found.")
    st.stop()

# ===== LOAD DATA =====
df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
df = df.sort_values("Date")

# ===== LOAD MODEL & SCALER =====
@st.cache_resource
def load_resources():
    m = load_model(MODEL_PATH, compile=False)
    with open(SCALER_PATH, "rb") as f:
        s = pickle.load(f)
    return m, s

model, scaler = load_resources()

# ===== DATASET STATS =====
avg_demand = df["Demand"].mean()
max_demand = df["Demand"].max()
min_demand = df["Demand"].min()
total_days = len(df)
date_range_str = f"{df['Date'].min().strftime('%b %Y')} — {df['Date'].max().strftime('%b %Y')}"

# Recent trend
recent_7 = df["Demand"].iloc[-7:].mean()
prev_7 = df["Demand"].iloc[-14:-7].mean()
trend_pct = ((recent_7 - prev_7) / prev_7 * 100)
trend_dir = "↑" if trend_pct > 0 else "↓"

st.markdown(f"""
<div class="stats-row">
    <div class="stat-item">
        <div class="stat-icon">📊</div>
        <div class="stat-value">{avg_demand:.0f} L</div>
        <div class="stat-label">Average Daily Demand</div>
        <div class="stat-sub">{date_range_str}</div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">📈</div>
        <div class="stat-value">{max_demand:.0f} L</div>
        <div class="stat-label">Peak Demand</div>
        <div class="stat-sub">Maximum recorded</div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">📉</div>
        <div class="stat-value">{min_demand:.0f} L</div>
        <div class="stat-label">Lowest Demand</div>
        <div class="stat-sub">Minimum recorded</div>
    </div>
    <div class="stat-item">
        <div class="stat-icon">🔄</div>
        <div class="stat-value">{trend_dir} {abs(trend_pct):.1f}%</div>
        <div class="stat-label">Weekly Trend</div>
        <div class="stat-sub">{total_days:,} data points</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===== TABS =====
tab1, tab2, tab3 = st.tabs(["🔮 Forecast", "📊 Historical Analysis", "📋 Data Explorer"])

# ===== TAB 1: FORECAST =====
with tab1:
    col_ctrl, col_chart = st.columns([1, 3])

    with col_ctrl:
        st.markdown("""
        <div class="glass-card">
            <div class="card-title">⚙️ Forecast Settings</div>
        </div>
        """, unsafe_allow_html=True)

        n_days = st.slider(
            "Forecast Days",
            min_value=3,
            max_value=30,
            value=7,
            help="Number of days to predict into the future"
        )

        seq_length = 10  # Fixed sequence length matching model training

        forecast_btn = st.button("🔮 Generate Forecast")

    with col_chart:
        if forecast_btn:
            # Prepare data for prediction
            demand = df["Demand"].values.reshape(-1, 1)
            scaled = scaler.transform(demand)

            # Forecast function
            def forecast_future(n_future, seq_len=10):
                last_seq = scaled[-seq_len:]
                preds = []

                for _ in range(n_future):
                    x = last_seq.reshape((1, seq_len, 1))
                    pred = model.predict(x, verbose=0)[0][0]
                    preds.append(pred)
                    last_seq = np.append(last_seq[1:], [[pred]], axis=0)

                return scaler.inverse_transform(
                    np.array(preds).reshape(-1, 1)
                ).flatten()

            future_preds = forecast_future(n_days, seq_length)

            future_dates = pd.date_range(
                df["Date"].iloc[-1] + pd.Timedelta(days=1),
                periods=n_days
            )

            forecast_df = pd.DataFrame({
                "Date": future_dates,
                "Predicted_Demand": np.round(future_preds, 1)
            })

            # ===== INTERACTIVE PLOTLY CHART =====
            history_days = min(180, len(df))
            history = df.tail(history_days)

            fig = go.Figure()

            # Historical line
            fig.add_trace(go.Scatter(
                x=history["Date"],
                y=history["Demand"],
                mode='lines',
                name='Historical Demand',
                line=dict(color='#00b4d8', width=2),
                fill='tozeroy',
                fillcolor='rgba(0, 180, 216, 0.08)'
            ))

            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast_df["Date"],
                y=forecast_df["Predicted_Demand"],
                mode='lines+markers',
                name='Forecasted Demand',
                line=dict(color='#ff8c00', width=3, dash='dot'),
                marker=dict(size=8, color='#ff8c00', line=dict(width=2, color='#ffffff'))
            ))

            # Confidence band (±10%)
            upper = forecast_df["Predicted_Demand"] * 1.10
            lower = forecast_df["Predicted_Demand"] * 0.90

            fig.add_trace(go.Scatter(
                x=pd.concat([forecast_df["Date"], forecast_df["Date"][::-1]]),
                y=pd.concat([upper, lower[::-1]]),
                fill='toself',
                fillcolor='rgba(255, 140, 0, 0.1)',
                line=dict(color='rgba(255,140,0,0)'),
                name='Confidence Band (±10%)',
                showlegend=True
            ))

            fig.update_layout(
                title=dict(
                    text=f"📈 Water Demand — {n_days}-Day Forecast",
                    font=dict(size=18, color='#ffffff')
                ),
                xaxis=dict(
                    title="Date",
                    gridcolor='rgba(255,255,255,0.1)',
                    color='#FFFFFF',
                    tickfont=dict(size=12, color='#FFFFFF', weight='bold')
                ),
                yaxis=dict(
                    title="Demand (Liters)",
                    gridcolor='rgba(255,255,255,0.1)',
                    color='#FFFFFF',
                    tickfont=dict(size=12, color='#FFFFFF', weight='bold')
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#FFFFFF', size=14),
                legend=dict(
                    bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#FFFFFF', size=12, weight='bold')
                ),
                height=450,
                margin=dict(l=20, r=20, t=60, b=20)
            )

            st.plotly_chart(fig, use_container_width=True)

            # ===== FORECAST TABLE =====
            st.markdown("#### 📋 Forecast Details")
            forecast_display = forecast_df.copy()
            forecast_display["Date"] = forecast_display["Date"].dt.strftime("%b %d, %Y (%A)")
            forecast_display.columns = ["Date", "Predicted Demand (L)"]
            st.dataframe(forecast_display, use_container_width=True, hide_index=True)

            # Summary stats
            st.markdown("")
            c1, c2, c3 = st.columns(3)
            c1.metric("📊 Avg Forecast", f"{future_preds.mean():.0f} L")
            c2.metric("📈 Max Forecast", f"{future_preds.max():.0f} L")
            c3.metric("📉 Min Forecast", f"{future_preds.min():.0f} L")

        else:
            st.info("👈 Configure forecast settings and click **Generate Forecast** to see predictions")


# ===== TAB 2: HISTORICAL ANALYSIS =====
with tab2:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📊 Historical Demand Analysis</div>
    </div>
    """, unsafe_allow_html=True)

    # Full timeline chart
    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=df["Date"],
        y=df["Demand"],
        mode='lines',
        name='Daily Demand',
        line=dict(color='#00b4d8', width=1.5),
        fill='tozeroy',
        fillcolor='rgba(0, 180, 216, 0.06)'
    ))

    # Monthly average overlay
    monthly = df.set_index("Date").resample("M")["Demand"].mean().reset_index()
    fig_full.add_trace(go.Scatter(
        x=monthly["Date"],
        y=monthly["Demand"],
        mode='lines+markers',
        name='Monthly Average',
        line=dict(color='#ff8c00', width=2.5),
        marker=dict(size=5)
    ))

    fig_full.update_layout(
        title="Full Historical Demand Timeline",
        xaxis_title="Date",
        yaxis_title="Demand (L)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family='Inter', color='#a0c4d8'),
        legend=dict(bgcolor='rgba(0,0,0,0)'),
        height=400,
        margin=dict(l=20, r=20, t=60, b=20)
    )
    fig_full.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
    fig_full.update_yaxes(gridcolor='rgba(255,255,255,0.05)')

    st.plotly_chart(fig_full, use_container_width=True)

    # Seasonal & Day-of-week patterns
    col1, col2 = st.columns(2)

    with col1:
        # Monthly boxplot
        df_temp = df.copy()
        df_temp["Month"] = df_temp["Date"].dt.month_name().str[:3]
        month_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                       "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        df_temp["Month"] = pd.Categorical(df_temp["Month"], categories=month_order, ordered=True)

        fig_month = px.box(
            df_temp, x="Month", y="Demand",
            title="Seasonal Demand Distribution",
            color_discrete_sequence=['#00b4d8']
        )
        fig_month.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#a0c4d8'),
            height=350,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_month.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig_month.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_month, use_container_width=True)

    with col2:
        # Day of week pattern
        if "Day_of_Week" in df.columns:
            dow_avg = df.groupby("Day_of_Week")["Demand"].mean().reset_index()
            day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            dow_avg["Day_Name"] = dow_avg["Day_of_Week"].map(lambda x: day_names[x])

            fig_dow = px.bar(
                dow_avg, x="Day_Name", y="Demand",
                title="Average Demand by Day of Week",
                color="Demand",
                color_continuous_scale=["#0077b6", "#00b4d8", "#00d4ff"]
            )
            fig_dow.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter', color='#a0c4d8'),
                height=350,
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=20, r=20, t=60, b=20)
            )
            fig_dow.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
            fig_dow.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
            st.plotly_chart(fig_dow, use_container_width=True)

    # Temperature vs Demand correlation
    if "Temperature" in df.columns:
        st.markdown("#### 🌡️ Temperature vs. Demand Correlation")
        fig_corr = px.scatter(
            df.sample(min(500, len(df)), random_state=42),
            x="Temperature", y="Demand",
            color="Demand",
            color_continuous_scale=["#0077b6", "#00b4d8", "#ff8c00"],
            opacity=0.6,
            trendline="ols",
            title="Temperature Impact on Water Demand"
        )
        fig_corr.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family='Inter', color='#a0c4d8'),
            height=380,
            coloraxis_showscale=False,
            margin=dict(l=20, r=20, t=60, b=20)
        )
        fig_corr.update_xaxes(gridcolor='rgba(255,255,255,0.05)')
        fig_corr.update_yaxes(gridcolor='rgba(255,255,255,0.05)')
        st.plotly_chart(fig_corr, use_container_width=True)


# ===== TAB 3: DATA EXPLORER =====
with tab3:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📋 Dataset Explorer</div>
    </div>
    """, unsafe_allow_html=True)

    # Data overview
    st.markdown("#### Dataset Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns)}")
    c3.metric("Date Range", f"{(df['Date'].max() - df['Date'].min()).days} days")
    c4.metric("Features", ", ".join([c for c in df.columns if c != 'Date']))

    st.markdown("#### 📄 Last 20 Records")
    st.dataframe(df.tail(20), use_container_width=True, hide_index=True)

    st.markdown("#### 📊 Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)


# ===== BACK NAVIGATION =====
st.markdown("---")
col1, col2, col3 = st.columns([3, 1, 3])
with col2:
    if st.button("⬅ Dashboard"):
        st.switch_page("pages/dashboard.py")
