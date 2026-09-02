import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AutoValue — Used Car Valuation",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# CUSTOM DESIGN
# ============================================================

st.html("""
<style>
.stApp {
    background: #f5f7fb;
}

.block-container {
    max-width: 1200px;
    padding-top: 30px;
    padding-bottom: 60px;
}

#MainMenu {
    visibility: hidden;
}

header {
    visibility: hidden;
}

footer {
    visibility: hidden;
}


/* HERO */

.hero {
    background: linear-gradient(135deg, #080b12 0%, #111827 55%, #1e293b 100%);
    border-radius: 28px;
    padding: 55px;
    margin-bottom: 35px;
    color: white;
    box-shadow: 0 25px 60px rgba(15, 23, 42, 0.20);
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,0.10);
    border: 1px solid rgba(255,255,255,0.14);
    color: #cbd5e1;
    padding: 8px 15px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
    margin-bottom: 20px;
}

.hero-title {
    font-size: 54px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 18px;
}

.hero-title span {
    color: #60a5fa;
}

.hero-description {
    max-width: 680px;
    color: #cbd5e1;
    font-size: 17px;
    line-height: 1.6;
}


/* SECTION */

.section-title {
    color: #0f172a;
    font-size: 26px;
    font-weight: 800;
    letter-spacing: -0.5px;
    margin-top: 25px;
}

.section-description {
    color: #64748b;
    font-size: 14px;
    margin-top: 5px;
    margin-bottom: 25px;
}


/* INPUTS */

.stSelectbox label,
.stNumberInput label {
    color: #334155 !important;
    font-weight: 700 !important;
    font-size: 13px !important;
}

.stSelectbox > div > div,
.stNumberInput > div > div {
    border-radius: 12px !important;
}


/* BUTTON */

.stButton {
    margin-top: 25px;
}

.stButton > button {
    width: 100% !important;
    height: 58px !important;
    border-radius: 14px !important;

    background: linear-gradient(135deg, #2563eb, #1d4ed8) !important;

    color: white !important;
    border: 1px solid #2563eb !important;

    font-size: 16px !important;
    font-weight: 800 !important;

    box-shadow: 0 10px 25px rgba(37, 99, 235, 0.22) !important;

    transition: all 0.18s ease !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8, #1e40af) !important;
    color: white !important;
    border: 1px solid #1d4ed8 !important;

    transform: translateY(-2px) !important;

    box-shadow: 0 15px 35px rgba(37, 99, 235, 0.35) !important;
}

.stButton > button:active {
    background: #1e40af !important;
    color: white !important;
    transform: translateY(0px) !important;
}


/* RESULT */

.result-card {
    background: linear-gradient(135deg, #080b12, #111827);
    border-radius: 24px;
    padding: 42px;
    margin-top: 40px;
    color: white;
    box-shadow: 0 25px 55px rgba(15, 23, 42, 0.22);
}

.result-label {
    color: #94a3b8;
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-price {
    color: white;
    font-size: 52px;
    font-weight: 900;
    letter-spacing: -2px;
    margin-top: 5px;
}

.result-subtitle {
    color: #94a3b8;
    font-size: 13px;
    margin-top: 5px;
}


/* SUMMARY */

.summary-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 18px;
    padding: 22px;
    min-height: 100px;
    box-shadow: 0 8px 25px rgba(15,23,42,0.04);
}

.summary-label {
    color: #64748b;
    font-size: 11px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.7px;
}

.summary-value {
    color: #0f172a;
    font-size: 18px;
    font-weight: 800;
    margin-top: 8px;
}


/* INFO */

.info-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 20px;
    margin-top: 22px;
    color: #64748b;
    font-size: 13px;
    line-height: 1.6;
}


/* FOOTER */

.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 12px;
    margin-top: 55px;
    padding-top: 25px;
    border-top: 1px solid #e2e8f0;
}
</style>
""")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model_files():

    model = joblib.load(
        BASE_DIR / "model" / "lasso_model.pkl"
    )

    scaler = joblib.load(
        BASE_DIR / "model" / "scaler.pkl"
    )

    feature_columns = joblib.load(
        BASE_DIR / "model" / "feature_columns.pkl"
    )

    return model, scaler, feature_columns


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_dataset():

    return pd.read_csv(
        BASE_DIR / "data" / "cardekho_dataset.csv"
    )


model, scaler, feature_columns = load_model_files()
data = load_dataset()


# ============================================================
# HERO
# ============================================================

st.html("""
<div class="hero">
    <div class="hero-badge">AI-POWERED VEHICLE VALUATION</div>

    <div class="hero-title">
        Auto<span>Value</span>
    </div>

    <div class="hero-description">
        Discover what your used car could be worth.
        Enter the vehicle specifications below and get
        an estimated market value powered by machine learning.
    </div>
</div>
""")


# ============================================================
# VEHICLE INFORMATION
# ============================================================

st.html("""
<div class="section-title">
    Vehicle information
</div>

<div class="section-description">
    Enter the specifications of the vehicle you want to value.
</div>
""")


# ============================================================
# INPUTS
# ============================================================

left, right = st.columns(2, gap="large")


with left:

    brand = st.selectbox(
        "Brand",
        sorted(data["brand"].dropna().unique())
    )

    available_models = sorted(
        data[data["brand"] == brand]["model"]
        .dropna()
        .unique()
    )

    model_name = st.selectbox(
        "Model",
        available_models
    )

    vehicle_age = st.number_input(
        "Vehicle age (years)",
        min_value=0,
        max_value=30,
        value=5,
        step=1
    )

    km_driven = st.number_input(
        "Kilometers driven",
        min_value=100,
        max_value=3800000,
        value=50000,
        step=1000
    )

    seats = st.number_input(
        "Number of seats",
        min_value=1,
        max_value=9,
        value=5,
        step=1
    )


with right:

    fuel_type = st.selectbox(
        "Fuel type",
        sorted(data["fuel_type"].dropna().unique())
    )

    transmission_type = st.selectbox(
        "Transmission",
        sorted(data["transmission_type"].dropna().unique())
    )

    seller_type = st.selectbox(
        "Seller type",
        sorted(data["seller_type"].dropna().unique())
    )

    mileage = st.number_input(
        "Mileage (km/l)",
        min_value=4.0,
        max_value=33.54,
        value=19.67,
        step=0.1
    )

    engine = st.number_input(
        "Engine displacement (cc)",
        min_value=793,
        max_value=6592,
        value=1200,
        step=50
    )

    max_power = st.number_input(
        "Maximum power (bhp)",
        min_value=38.4,
        max_value=626.0,
        value=88.5,
        step=1.0
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

predict = st.button(
    "🚘   Estimate Market Value",
    use_container_width=True
)


# ============================================================
# PREDICTION
# ============================================================

if predict:

    input_data = pd.DataFrame({
        "vehicle_age": [vehicle_age],
        "km_driven": [km_driven],
        "seller_type": [seller_type],
        "fuel_type": [fuel_type],
        "transmission_type": [transmission_type],
        "mileage": [mileage],
        "engine": [engine],
        "max_power": [max_power],
        "seats": [seats],
        "brand": [brand],
        "model": [model_name]
    })


    # Same encoding used during training

    input_data = pd.get_dummies(
        input_data,
        columns=[
            "brand",
            "model",
            "seller_type",
            "fuel_type",
            "transmission_type"
        ],
        drop_first=True
    )


    # Same 163 features and same order

    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )


    # Same scaler used during training

    input_scaled = scaler.transform(
        input_data
    )


    # Model predicts log(price)

    prediction_log = model.predict(
        input_scaled
    )


    # Convert log(price) back to rupees

    predicted_price = np.expm1(
        prediction_log[0]
    )

    predicted_price = max(
        predicted_price,
        0
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.html(f"""
<div class="result-card">
    <div class="result-label">
        Estimated market value
    </div>

    <div class="result-price">
        ₹{predicted_price:,.0f}
    </div>

    <div class="result-subtitle">
        Machine-learning estimate based on the vehicle specifications provided.
    </div>
</div>
""")


    # ========================================================
    # SUMMARY
    # ========================================================

    st.html("""
<div class="section-title">
    Valuation summary
</div>

<div class="section-description">
    Vehicle characteristics used for this estimate.
</div>
""")


    summary1, summary2, summary3 = st.columns(3)


    with summary1:

        st.html(f"""
<div class="summary-card">
    <div class="summary-label">
        Vehicle
    </div>

    <div class="summary-value">
        {brand} {model_name}
    </div>
</div>
""")


    with summary2:

        st.html(f"""
<div class="summary-card">
    <div class="summary-label">
        Vehicle age
    </div>

    <div class="summary-value">
        {vehicle_age} years
    </div>
</div>
""")


    with summary3:

        st.html(f"""
<div class="summary-card">
    <div class="summary-label">
        Distance driven
    </div>

    <div class="summary-value">
        {km_driven:,} km
    </div>
</div>
""")


    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.html("""
<div class="info-card">
    🤖 <strong>Machine-learning valuation</strong><br>
    This estimate is generated using the trained Tuned Log
    Lasso regression model developed from the CarDekho dataset.
    Actual selling prices can vary depending on vehicle condition,
    location, ownership history and market demand.
</div>
""")


# ============================================================
# FOOTER
# ============================================================

st.html("""
<div class="footer">
    <strong>AutoValue</strong> · Used Car Price Prediction<br>
    Machine Learning Project · Tuned Log Lasso Regression
</div>
""")