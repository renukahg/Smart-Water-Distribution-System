import streamlit as st
import json
import os
import re
import hashlib

# ---------------- CONFIG ----------------
USER_FILE = "users.json"

st.set_page_config(
    page_title="AquaFlow Intelligence | Login",
    page_icon="💧",
    layout="centered"
)

# ---------------- LOAD / INIT USERS ----------------
if not os.path.exists(USER_FILE):
    with open(USER_FILE, "w") as f:
        json.dump({"admin": "Admin@123"}, f)

with open(USER_FILE, "r") as f:
    users = json.load(f)

# ---------------- SESSION STATE ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "show_register" not in st.session_state:
    st.session_state.show_register = False

if "show_forgot" not in st.session_state:
    st.session_state.show_forgot = False

# ---------------- PREMIUM CSS ----------------
st.markdown("""
<style>
    /* ===== IMPORT GOOGLE FONT ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ===== ANIMATED GRADIENT BACKGROUND ===== */
    .stApp {
        background: linear-gradient(-45deg, #0a1628, #0d2137, #0a2a4a, #051a2e, #0c1929);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        font-family: 'Inter', sans-serif;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* ===== HIDE STREAMLIT DEFAULTS ===== */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    section[data-testid="stSidebar"] { display: none; }

    /* ===== ANIMATED WATER PARTICLES ===== */
    .water-bg {
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        pointer-events: none;
        z-index: 0;
        overflow: hidden;
    }

    .water-bg::before, .water-bg::after {
        content: '';
        position: absolute;
        width: 300px; height: 300px;
        border-radius: 40%;
        opacity: 0.06;
        animation: waterFloat 8s ease-in-out infinite;
    }

    .water-bg::before {
        background: linear-gradient(135deg, #00d4ff, #0098db);
        top: -50px; right: -50px;
        animation-delay: 0s;
    }

    .water-bg::after {
        background: linear-gradient(135deg, #00b4d8, #0077b6);
        bottom: -80px; left: -80px;
        animation-delay: 4s;
        width: 400px; height: 400px;
    }

    @keyframes waterFloat {
        0%, 100% { transform: rotate(0deg) scale(1); }
        25% { transform: rotate(90deg) scale(1.1); }
        50% { transform: rotate(180deg) scale(1); }
        75% { transform: rotate(270deg) scale(1.05); }
    }

    /* ===== MAIN CONTAINER ===== */
    .login-wrapper {
        max-width: 440px;
        margin: 0 auto;
        padding: 20px 0;
        position: relative;
        z-index: 1;
    }

    .stApp {
        background: linear-gradient(135deg, #001a2c 0%, #004d7a 50%, #001a2c 100%);
        font-family: 'Inter', sans-serif;
    }

    .brand-icon {
        width: 80px; height: 80px;
        background: linear-gradient(135deg, #00d4ff 0%, #0077b6 100%);
        border-radius: 24px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 40px;
        margin-bottom: 16px;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3);
        animation: iconPulse 3s ease-in-out infinite;
    }

    @keyframes iconPulse {
        0%, 100% { box-shadow: 0 8px 32px rgba(0, 212, 255, 0.3); }
        50% { box-shadow: 0 8px 48px rgba(0, 212, 255, 0.5); }
    }

    .brand-title {
        font-size: 36px;
        font-weight: 950;
        background: linear-gradient(135deg, #ffffff, #00f2ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 4px;
        letter-spacing: -1.5px;
        text-shadow: 0 0 30px rgba(0, 242, 255, 0.7);
    }

    .brand-subtitle {
        font-size: 16px;
        color: #00f2ff;
        font-weight: 900;
        letter-spacing: 4px;
        text-transform: uppercase;
        text-shadow: 0 0 15px rgba(0, 242, 255, 0.5);
    }

    /* ===== GLASSMORPHISM CARD ===== */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 40px 36px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }

    .glass-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        right: 0; height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    }

    .card-title {
        font-size: 22px;
        font-weight: 700;
        color: #ffffff;
        margin-bottom: 6px;
    }

    .card-desc {
        font-size: 15px;
        color: #FFFFFF !important;
        margin-bottom: 24px;
        font-weight: 800;
        text-shadow: 0 0 8px rgba(255, 255, 255, 0.3);
    }

    /* ===== FORM INPUTS ===== */
    .stTextInput input {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
        background-color: rgba(255, 255, 255, 0.95) !important;
    }

    .stTextInput > div > div > input {
        background: rgba(255, 255, 255, 0.95) !important;
        border: 1px solid rgba(255, 255, 255, 0.3) !important;
        border-radius: 14px !important;
        color: #000000 !important;
        padding: 14px 18px !important;
        font-size: 15px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    .stTextInput > div > div > input:focus {
        border-color: #00d4ff !important;
        box-shadow: 0 0 0 3px rgba(0, 212, 255, 0.15) !important;
        background: rgba(255, 255, 255, 0.1) !important;
    }

    .stTextInput > div > div > input::placeholder {
        color: #A0A0A0 !important;
    }

    .stTextInput > label {
        color: #FFFFFF !important;
        font-size: 14px !important;
        font-weight: 800 !important;
        letter-spacing: 1px !important;
        text-transform: uppercase;
        text-shadow: 0 0 5px rgba(255,255,255,0.2);
    }

    /* ===== PRIMARY BUTTON ===== */
    .stFormSubmitButton > button,
    div.stButton > button:first-child {
        width: 100%;
        background: linear-gradient(135deg, #00b4d8 0%, #0077b6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 14px !important;
        padding: 14px 24px !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        font-family: 'Inter', sans-serif !important;
        letter-spacing: 0.5px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 180, 216, 0.3) !important;
    }

    .stFormSubmitButton > button:hover,
    div.stButton > button:first-child:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(0, 180, 216, 0.5) !important;
        background: linear-gradient(135deg, #00c6e0 0%, #0088cc 100%) !important;
    }

    /* ===== SECONDARY / LINK BUTTONS ===== */
    .link-btn {
        text-align: center;
        margin-top: 16px;
    }

    .link-btn a, .link-btn span {
        color: #00F2FF;
        font-size: 13px;
        font-weight: 600;
        cursor: pointer;
    }

    /* ===== DIVIDER ===== */
    .divider {
        display: flex;
        align-items: center;
        margin: 20px 0;
        color: #FFFFFF;
        font-size: 12px;
    }

    .divider::before, .divider::after {
        content: '';
        flex: 1;
        height: 1px;
        background: rgba(255,255,255,0.1);
    }

    .divider span {
        padding: 0 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #FFFFFF;
        font-weight: 600;
    }

    /* ===== FOOTER ===== */
    .login-footer {
        text-align: center;
        margin-top: 24px;
        font-size: 12px;
        color: #FFFFFF;
        font-weight: 500;
    }

    .login-footer a {
        color: #00F2FF;
        text-decoration: none;
        font-weight: 600;
    }

    /* ===== FEATURE PILLS ===== */
    .features {
        display: flex;
        justify-content: center;
        gap: 12px;
        margin-top: 20px;
        flex-wrap: wrap;
    }

    .feature-pill {
        background: rgba(0, 212, 255, 0.1);
        border: 1px solid rgba(0, 212, 255, 0.2);
        border-radius: 20px;
        padding: 6px 16px;
        font-size: 11px;
        color: #64d8ff;
        letter-spacing: 0.5px;
    }

    /* ===== ALERTS ===== */
    .stAlert {
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* ===== PASSWORD STRENGTH BAR ===== */
    .strength-bar {
        height: 4px;
        border-radius: 2px;
        margin-top: 8px;
        transition: all 0.3s ease;
    }

    .strength-weak { background: linear-gradient(90deg, #ff4444 0%, #ff4444 33%, #333 33%); }
    .strength-medium { background: linear-gradient(90deg, #ffaa00 0%, #ffaa00 66%, #333 66%); }
    .strength-strong { background: linear-gradient(90deg, #00cc66 0%, #00cc66 100%); }
</style>

<div class="water-bg"></div>
""", unsafe_allow_html=True)


# ===== PASSWORD VALIDATION =====
def validate_password(pw):
    if len(pw) < 8:
        return "⚠️ Minimum 8 characters required"
    if not re.search(r"[A-Z]", pw):
        return "⚠️ At least one uppercase letter required"
    if not re.search(r"[a-z]", pw):
        return "⚠️ At least one lowercase letter required"
    if not re.search(r"\d", pw):
        return "⚠️ At least one number required"
    if not re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", pw):
        return "⚠️ At least one special character required"
    return None


def get_password_strength(pw):
    if len(pw) == 0:
        return "", ""
    score = 0
    if len(pw) >= 8: score += 1
    if re.search(r"[A-Z]", pw): score += 1
    if re.search(r"[a-z]", pw): score += 1
    if re.search(r"\d", pw): score += 1
    if re.search(r"[!@#$%^&*()_+\-=\[\]{}|;:,.<>?]", pw): score += 1

    if score <= 2:
        return "Weak", "strength-weak"
    elif score <= 3:
        return "Medium", "strength-medium"
    else:
        return "Strong", "strength-strong"


# ===== BRAND HEADER =====
st.markdown("""
<div class="login-wrapper">
    <div class="brand-section">
        <div class="brand-icon">💧</div>
        <div class="brand-title">AquaFlow Intelligence</div>
        <div class="brand-subtitle">Smart Water Distribution System</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ===================================================================
# FORGOT PASSWORD MODE
# ===================================================================
if st.session_state.show_forgot:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">🔑 Reset Password</div>
        <div class="card-desc">Enter your username and set a new password</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("forgot_form"):
        fp_user = st.text_input("Username", placeholder="Enter your username")
        fp_new = st.text_input("New Password", type="password", placeholder="Enter new password")
        fp_confirm = st.text_input("Confirm Password", type="password", placeholder="Confirm new password")
        reset_btn = st.form_submit_button("🔄 Reset Password")

    if reset_btn:
        if not fp_user or not fp_new:
            st.error("All fields are required")
        elif fp_user.strip().lower() not in users:
            st.error("❌ Username not found")
        elif fp_new != fp_confirm:
            st.error("❌ Passwords do not match")
        else:
            error = validate_password(fp_new)
            if error:
                st.error(error)
            else:
                users[fp_user.strip().lower()] = fp_new
                with open(USER_FILE, "w") as f:
                    json.dump(users, f)
                st.success("✅ Password reset successful! Please login.")
                st.session_state.show_forgot = False
                st.rerun()

    if st.button("⬅ Back to Login"):
        st.session_state.show_forgot = False
        st.session_state.show_register = False
        st.rerun()

# ===================================================================
# REGISTER MODE
# ===================================================================
elif st.session_state.show_register:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">📝 Create Account</div>
        <div class="card-desc">Join AquaFlow Intelligence to monitor your water network</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("register_form"):
        new_username = st.text_input("Username", placeholder="Choose a username (min 6 chars)")
        new_password = st.text_input("New Password", type="password", placeholder="Create a strong password")
        confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
        register_btn = st.form_submit_button("🚀 Create Account")

    if register_btn:
        clean_user = new_username.strip().lower()
        if not clean_user or not new_password:
            st.error("All fields are required")
        elif len(clean_user) < 6:
            st.error("⚠️ Username must be at least 6 characters")
        elif clean_user in users:
            st.error("❌ Username already exists — try a different one")
        elif new_password != confirm_password:
            st.error("❌ Passwords do not match")
        else:
            error = validate_password(new_password)
            if error:
                st.error(error)
            else:
                users[clean_user] = new_password
                with open(USER_FILE, "w") as f:
                    json.dump(users, f)
                st.success("✅ Account created successfully! Please login.")
                st.session_state.show_register = False
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back to Login"):
            st.session_state.show_register = False
            st.rerun()

# ===================================================================
# LOGIN MODE (DEFAULT)
# ===================================================================
else:
    st.markdown("""
    <div class="glass-card">
        <div class="card-title">👋 Welcome Back</div>
        <div class="card-desc">Sign in to access your water management dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username", placeholder="Enter your username")
        password = st.text_input("Password", type="password", placeholder="Enter your password")
        login_btn = st.form_submit_button("🔐 Sign In")

    if login_btn:
        clean_user = username.strip().lower()
        if not clean_user or not password:
            st.error("Please enter both username and password")
        elif clean_user in users and users[clean_user] == password:
            st.session_state.logged_in = True
            st.session_state.user = clean_user
            st.success("✅ Login successful! Redirecting...")
            st.switch_page("pages/dashboard.py")
        else:
            st.error("❌ Invalid username or password")

    st.markdown('<div class="divider"><span>or</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📝 Create Account"):
            st.session_state.show_register = True
            st.rerun()
    with col2:
        if st.button("🔑 Forgot Password"):
            st.session_state.show_forgot = True
            st.rerun()

# ===== FEATURE PILLS =====
st.markdown("""
<div class="features">
    <span class="feature-pill">🔍 Leak Detection</span>
    <span class="feature-pill">📈 Demand Forecasting</span>
    <span class="feature-pill">📊 Real-time Analytics</span>
</div>
""", unsafe_allow_html=True)

# ===== FOOTER =====
st.markdown("""
<div class="login-footer">
    <p>© 2025 AquaFlow Intelligence — Smart Water Distribution System</p>
    <p>Powered by Machine Learning & LSTM Deep Learning</p>
</div>
""", unsafe_allow_html=True)
# Additional Ultra-Bright Styles
st.markdown("""
<style>
    h1, h2, h3, h4 {
        color: #FFFFFF !important;
        font-weight: 950 !important;
        text-shadow: 0 0 15px rgba(255,255,255,0.4) !important;
    }
    div[data-testid="stMetricValue"] {
        color: #FFFFFF !important;
        font-weight: 900 !important;
        text-shadow: 0 0 10px rgba(0, 242, 255, 0.4);
    }
    div[data-testid="stMetricLabel"] {
        color: #00F2FF !important;
        font-weight: 800 !important;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)
