import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# ---------- Page Config ----------
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom Styling ----------
st.markdown(
    """
    <style>
        :root {
            --card-bg: rgba(255,255,255,0.75);
            --card-border: rgba(0,0,0,0.06);
            --text: #111827;
        }
        @media (prefers-color-scheme: dark) {
            :root {
                --card-bg: rgba(30,41,59,0.55);
                --card-border: rgba(255,255,255,0.08);
                --text: #e5e7eb;
            }
        }
        .metric-card {
            border-radius: 16px;
            padding: 16px 18px;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            backdrop-filter: blur(8px);
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }
        .metric-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 10px 24px rgba(0,0,0,0.12);
        }
        .metric-emoji {
            font-size: 22px; line-height: 1;
        }
        .metric-title {
            font-size: 12px; opacity: 0.8; margin-top: 6px;
        }
        .metric-value {
            font-weight: 700; font-size: 18px; color: var(--text); margin-top: 2px;
            word-break: break-word;
        }
        .stMetric {
            background-color: #1e1e2f;
            padding: 20px; border-radius: 16px; color: white;
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        .stProgress > div > div { border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Load Artifacts ----------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = os.path.join("models", "model.keras")
    encoders_path = os.path.join("models", "encoders.pkl")
    scaler_path = os.path.join("models", "scaler.pkl")

    model = tf.keras.models.load_model(model_path)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    return model, encoders, scaler

try:
    model, encoders, scaler = load_artifacts()
    gender_encoder = encoders.get("gender_encoder")
    geo_encoder = encoders.get("geo_encoder")
except Exception as e:
    st.error("⚠️ Model or encoders not found. Please ensure `models/model.keras`, `models/encoders.pkl`, and `models/scaler.pkl` exist.")
    st.stop()
# ---------- Helpers ----------
NUMERIC_ORDER = [
    "CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
]
RAW_ORDER = [
    "CreditScore","Geography","Gender","Age","Tenure","Balance","NumOfProducts",
    "HasCrCard","IsActiveMember","EstimatedSalary"
]

def preprocess(df):
    df = df.copy()

    # Encode Gender
    if gender_encoder is not None:
        try:
            df["Gender"] = gender_encoder.transform(df["Gender"])
        except Exception:
            df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    else:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Encode Geography
    if geo_encoder is not None:
        geo = geo_encoder.transform(df[["Geography"]])
        try:
            geo = geo.toarray()
        except Exception:
            pass
        geo_cols = getattr(geo_encoder, "get_feature_names_out", lambda f: [f"Geography_{c}" for c in geo_encoder.categories_[0]])(["Geography"])
        geo_df = pd.DataFrame(geo, columns=geo_cols, index=df.index)
        df = pd.concat([df.drop("Geography", axis=1), geo_df], axis=1)
    else:
        df = pd.concat([df.drop(columns=["Geography"]), pd.get_dummies(df["Geography"], prefix="Geography")], axis=1)

    # Ensure numeric types
    for c in ["HasCrCard","IsActiveMember","NumOfProducts","Tenure"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
    for c in ["CreditScore","Age","Balance","EstimatedSalary"]:
        if c in df: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # Align to scaler features if available
    if hasattr(scaler, "feature_names_in_"):
        need = list(scaler.feature_names_in_)
        for col in need:
            if col not in df.columns: df[col] = 0
        X = df[need].copy()
    else:
        X = df.copy()

    X_scaled = scaler.transform(X)
    return X_scaled

def predict(df):
    X = preprocess(df)
    prob = model.predict(X, verbose=0).ravel()
    # If model returns 2-class softmax
    if prob.ndim > 1:
        prob = prob[:, 1]
    return prob

def profile_cards(row: pd.Series):
    """Render a responsive grid of profile cards with emojis."""
    # Data map: (emoji, title, value)
    items = [
        ("📊", "Credit Score", f"{int(row.CreditScore):,}"),
        ("🌍", "Geography", str(row.Geography)),
        ("👤", "Gender", str(row.Gender)),
        ("🎂", "Age", f"{int(row.Age)}"),
        ("📅", "Tenure", f"{int(row.Tenure)} yrs"),
        ("🏦", "Balance", f"{row.Balance:,.2f}"),
        ("🧺", "Products", f"{int(row.NumOfProducts)}"),
        ("💳", "Has Card", "Yes" if int(row.HasCrCard) else "No"),
        ("⚡", "Active Member", "Yes" if int(row.IsActiveMember) else "No"),
        ("💵", "Est. Salary", f"{row.EstimatedSalary:,.2f}"),
    ]
    # Render in a 5x2 grid (responsive)
    for i in range(0, len(items), 5):
        cols = st.columns(5)
        for col, (emoji, title, value) in zip(cols, items[i:i+5]):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-emoji">{emoji}</div>
                        <div class="metric-title">{title}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ---------- Sidebar Inputs ----------
with st.sidebar:
    st.title("🔧 Configure Inputs")
    st.markdown("Enter customer details to predict churn:")
    CreditScore = st.slider("Credit Score", 300, 900, 650)
    Geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.slider("Age", 18, 92, 40)
    Tenure = st.slider("Tenure (Years)", 0, 10, 5)
    Balance = st.number_input("Balance", 0.0, 1_000_000.0, 100_000.0, step=1000.0, format="%.2f")
    NumOfProducts = st.slider("Num Of Products", 1, 4, 1)
    HasCrCard = st.checkbox("Has Credit Card", True)
    IsActiveMember = st.checkbox("Is Active Member", True)
    EstimatedSalary = st.number_input("Estimated Salary", 0.0, 1_000_000.0, 50_000.0, step=1000.0, format="%.2f")

    single_df = pd.DataFrame([{
        "CreditScore": CreditScore,
        "Geography": Geography,
        "Gender": Gender,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": int(HasCrCard),
        "IsActiveMember": int(IsActiveMember),
        "EstimatedSalary": EstimatedSalary
    }], columns=RAW_ORDER)

# ---------- Tabs Layout ----------
tab1, tab2 = st.tabs(["👤 Single Prediction", "📂 Batch Upload"])

# ---------- Single Prediction Tab ----------
with tab1:
    st.header("👤 Single Customer Prediction")
    # Profile cards first
    profile_cards(single_df.iloc[0])

    st.markdown("")

    # Action row
    colA, colB = st.columns([1, 2])
    with colA:
        if st.button("🔮 Predict Churn", use_container_width=True):
            prob = float(predict(single_df)[0])
            label = "Exited (Churn)" if prob >= 0.5 else "Retained (Stay)"
            st.metric("Churn Probability", f"{prob:.2%}")
            st.progress(min(max(prob, 0.0), 1.0))
            st.success(f"Prediction: **{label}**")

            st.markdown("### 📊 Risk Breakdown")
            fig = px.pie(
                names=["Stay", "Exit"],
                values=[1 - prob, prob],
                hole=0.45
            )
            st.plotly_chart(fig, use_container_width=True)

    with colB:
        st.markdown("### 📋 Customer Data")
        st.dataframe(single_df, use_container_width=True)

# ---------- Batch Upload Tab ----------
with tab2:
    st.header("📂 Batch Predictions")
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        df = pd.read_csv(file)
        # Ensure column order exists, ignore 'Exited' if present
        expected = [c for c in RAW_ORDER]
        missing = [c for c in expected if c not in df.columns]
        if missing:
            st.error(f"Missing required columns: {missing}")
        else:
            probs = predict(df[expected])
            df["Churn Probability"] = probs
            df["Prediction"] = np.where(probs >= 0.5, "Exited", "Retained")

            st.success(f"✅ Processed {len(df)} records")
            st.dataframe(df.head(50), use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")

            st.subheader("📊 Churn Probability Distribution")
            fig = px.histogram(df, x="Churn Probability", nbins=20, color="Prediction")
            st.plotly_chart(fig, use_container_width=True)

# ---------- Footer ----------
st.markdown(
    """
    ---
    🌟 **Customer Churn Predictor**
    """
)