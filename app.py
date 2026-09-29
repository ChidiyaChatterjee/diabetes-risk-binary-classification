import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "diabetes_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "diabetes_scaler.pkl")

FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# These are the medians used by the original project preprocessing.
# Keep these synchronized with the preprocessing used during training.
IMPUTATION_VALUES = {
    "Glucose": 117.0,
    "BloodPressure": 72.0,
    "SkinThickness": 23.0,
    "Insulin": 30.5,
    "BMI": 32.0,
}


# ============================================================
# STYLING
# ============================================================

st.markdown(
    """
    <style>

        /* =====================================================
           60-30-10 COLOR SYSTEM
           60% - #F8FAFC  Dominant
           30% - #0F172A  Secondary
           10% - #2563EB  Accent
           ===================================================== */

        :root {
            --dominant: #F8FAFC;
            --secondary: #0F172A;
            --accent: #2563EB;

            --white: #FFFFFF;
            --border: #E2E8F0;
            --muted: #64748B;

            --success: #16A34A;
            --warning: #F59E0B;
            --danger: #DC2626;
        }


        /* =====================================================
           GLOBAL
           ===================================================== */

        .stApp {
            background: var(--dominant);
            color: var(--secondary);
        }

        .block-container {
            max-width: 1180px;
            padding: 2rem 1.25rem 3rem;
        }


        /* =====================================================
           HERO - SECONDARY 30%
           ===================================================== */

        .hero {
            background: var(--secondary);
            color: var(--white);

            padding: 2.75rem;
            border-radius: 24px;

            margin-bottom: 1.5rem;

            box-shadow:
                0 16px 40px rgba(15, 23, 42, 0.14);
        }

        .hero h1 {
            margin: 0;

            color: var(--white);

            font-size: clamp(
                2rem,
                5vw,
                3.4rem
            );

            font-weight: 800;

            letter-spacing: -0.045em;
        }

        .hero p {
            max-width: 760px;

            margin: 0.8rem 0 0;

            color: #CBD5E1;

            font-size: 1.05rem;
            line-height: 1.7;
        }


        /* =====================================================
           CONTENT CARDS - DOMINANT 60%
           ===================================================== */

        .section-card {
            background: var(--white);

            border: 1px solid var(--border);

            border-radius: 20px;

            padding: 1.5rem;

            margin-bottom: 1.25rem;

            box-shadow:
                0 8px 28px rgba(15, 23, 42, 0.05);
        }

        .section-title {
            color: var(--secondary);

            font-size: 1.25rem;
            font-weight: 750;

            margin-bottom: 0.3rem;
        }

        .section-description {
            color: var(--muted);

            font-size: 0.92rem;

            margin-bottom: 1.25rem;
        }


        /* =====================================================
           INPUTS
           ===================================================== */

        div[data-testid="stNumberInput"] label {
            color: var(--secondary);

            font-weight: 650;
        }

        div[data-testid="stNumberInput"] input {
            background: var(--white);

            border: 1px solid var(--border);

            border-radius: 10px;

            color: var(--secondary);
        }

        div[data-testid="stNumberInput"] input:focus {
            border-color: var(--accent);

            box-shadow:
                0 0 0 1px var(--accent);
        }


        /* =====================================================
           PRIMARY BUTTON - 10% ACCENT
           ===================================================== */

        .stButton > button {
            width: 100%;

            background: var(--accent);
            color: var(--white);

            border: none;

            border-radius: 12px;

            padding: 0.8rem 1rem;

            font-size: 1rem;
            font-weight: 700;

            transition:
                transform 0.15s ease,
                box-shadow 0.15s ease,
                background 0.15s ease;
        }

        .stButton > button:hover {
            background: #1D4ED8;

            transform: translateY(-1px);

            box-shadow:
                0 8px 20px rgba(37, 99, 235, 0.25);
        }

        .stButton > button:active {
            transform: translateY(0);
        }


        /* =====================================================
           RESULT CARDS
           ===================================================== */

        .result-card {
            background: var(--white);

            border-radius: 20px;

            padding: 1.5rem;

            margin-top: 1.25rem;

            border: 1px solid var(--border);

            box-shadow:
                0 8px 28px rgba(15, 23, 42, 0.06);
        }

        .result-positive {
            border-left: 5px solid var(--danger);
        }

        .result-negative {
            border-left: 5px solid var(--success);
        }

        .result-title {
            color: var(--secondary);

            font-size: 1.6rem;
            font-weight: 800;

            margin-bottom: 0.35rem;
        }

        .result-text {
            color: var(--muted);

            margin: 0;

            line-height: 1.6;
        }

        .probability {
            color: var(--accent);

            font-size: 2.5rem;
            font-weight: 850;

            margin-top: 0.8rem;
        }


        /* =====================================================
           DISCLAIMER
           ===================================================== */

        .disclaimer {
            background: var(--white);

            border: 1px solid var(--border);

            border-radius: 16px;

            padding: 1rem 1.15rem;

            margin-top: 1.5rem;

            color: var(--muted);

            font-size: 0.82rem;

            line-height: 1.65;
        }

        .disclaimer strong {
            color: var(--secondary);
        }


        /* =====================================================
           FOOTER
           ===================================================== */

        .footer {
            text-align: center;

            color: var(--muted);

            font-size: 0.8rem;

            margin-top: 2rem;
        }


        /* =====================================================
           STREAMLIT ELEMENT REFINEMENT
           ===================================================== */

        div[data-testid="stAlert"] {
            border-radius: 12px;
        }

        div[data-testid="stExpander"] {
            border-radius: 14px;

            border: 1px solid var(--border);
        }


        /* =====================================================
           RESPONSIVE
           ===================================================== */

        @media (max-width: 768px) {

            .block-container {
                padding:
                    1.25rem
                    0.85rem
                    2rem;
            }

            .hero {
                padding: 1.75rem;

                border-radius: 20px;
            }

            .hero h1 {
                font-size: 2.1rem;
            }

            .hero p {
                font-size: 0.95rem;
            }

            .section-card {
                padding: 1.1rem;

                border-radius: 17px;
            }

            .result-card {
                padding: 1.15rem;
            }

            .probability {
                font-size: 2rem;
            }
        }


        @media (max-width: 480px) {

            .hero {
                padding: 1.4rem;
            }

            .hero h1 {
                font-size: 1.8rem;
            }

            .section-title {
                font-size: 1.1rem;
            }
        }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler file not found: {SCALER_PATH}"
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    return model, scaler


# ============================================================
# VALIDATION
# ============================================================

def validate_inputs(values):
    errors = []

    if values["Pregnancies"] < 0:
        errors.append("Pregnancies cannot be negative.")

    if values["Glucose"] <= 0:
        errors.append("Glucose must be greater than 0.")

    if values["BloodPressure"] <= 0:
        errors.append("Blood Pressure must be greater than 0.")

    if values["SkinThickness"] <= 0:
        errors.append("Skin Thickness must be greater than 0.")

    if values["Insulin"] <= 0:
        errors.append("Insulin must be greater than 0.")

    if values["BMI"] <= 0:
        errors.append("BMI must be greater than 0.")

    if values["DiabetesPedigreeFunction"] < 0:
        errors.append(
            "Diabetes Pedigree Function cannot be negative."
        )

    if values["Age"] <= 0:
        errors.append("Age must be greater than 0.")

    return errors


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_input(values):
    data = pd.DataFrame([values], columns=FEATURES)

    zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI",
    ]

    for column in zero_as_missing:
        if data.loc[0, column] == 0:
            data.loc[0, column] = IMPUTATION_VALUES[column]

    return data


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <h1>Diabetes Risk Prediction</h1>
        <p>
            A machine-learning based prediction interface using the trained
            Logistic Regression model from this project.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# INPUT FORM
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <div class="section-title">Patient Information</div>
        <div class="section-description">
            Enter the required clinical measurements below.
        </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=50,
        value=1,
        step=1,
    )

with col2:
    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=500.0,
        value=120.0,
        step=1.0,
    )

with col3:
    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=250.0,
        value=70.0,
        step=1.0,
    )

with col4:
    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=150.0,
        value=20.0,
        step=1.0,
    )

col5, col6, col7, col8 = st.columns(4)

with col5:
    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=1000.0,
        value=80.0,
        step=1.0,
    )

with col6:
    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=0.1,
    )

with col7:
    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01,
        format="%.2f",
    )

with col8:
    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1,
    )

st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# PREDICTION
# ============================================================

predict_clicked = st.button(
    "🔍 Predict Diabetes Risk",
    type="primary",
    use_container_width=True,
)


if predict_clicked:

    input_values = {
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }

    errors = validate_inputs(input_values)

    if errors:
        for error in errors:
            st.error(error)
        st.stop()

    try:
        with st.spinner("Running prediction..."):

            model, scaler = load_artifacts()

            input_df = preprocess_input(input_values)

            scaled_input = scaler.transform(input_df)

            prediction = int(model.predict(scaled_input)[0])

            probability = None

            if hasattr(model, "predict_proba"):
                probability = float(
                    model.predict_proba(scaled_input)[0][1]
                )

        if prediction == 1:

            probability_text = (
                f"{probability * 100:.1f}%"
                if probability is not None
                else "N/A"
            )

            st.markdown(
                f"""
                <div class="result-card result-positive">
                    <div class="result-title">
                        Diabetes Predicted
                    </div>
                    <p class="result-text">
                        The trained model predicts the positive class
                        for the supplied input values.
                    </p>
                    <div class="probability">
                        {probability_text}
                    </div>
                    <p class="result-text">
                        Model-estimated probability of the positive class
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:

            probability_text = (
                f"{probability * 100:.1f}%"
                if probability is not None
                else "N/A"
            )

            st.markdown(
                f"""
                <div class="result-card result-negative">
                    <div class="result-title">
                        No Diabetes Predicted
                    </div>
                    <p class="result-text">
                        The trained model predicts the negative class
                        for the supplied input values.
                    </p>
                    <div class="probability">
                        {probability_text}
                    </div>
                    <p class="result-text">
                        Model-estimated probability of the positive class
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    except FileNotFoundError as error:
        st.error(str(error))

    except Exception:
        st.error(
            "Prediction could not be completed. "
            "Please verify the model files and input values."
        )


# ============================================================
# DISCLAIMER
# ============================================================

st.markdown(
    """
    <div class="disclaimer">
        <strong>Important:</strong>
        This application is intended for educational and research purposes.
        It is not a medical diagnostic system and should not be used as a
        substitute for professional medical advice, diagnosis, or treatment.
        Model predictions may be affected by data quality, preprocessing,
        training-data limitations, and model performance.
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="footer">
        Diabetes Prediction ML Project
    </div>
    """,
    unsafe_allow_html=True,
)