import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import plotly.express as px
import os
from tensorflow.keras.models import load_model

# =============================
# CONFIG
# =============================
API_URL = "http://127.0.0.1:8000"
MODEL_PATH = "models/cnn_lstm_anomaly_detector.h5"
EXPECTED_FEATURES = 29

# =============================
# LOAD LOCAL MODEL (for fallback)
# =============================
@st.cache_resource
def load_local_model():
    return load_model(MODEL_PATH)

# =============================
# PAGE HEADER
# =============================
st.set_page_config(page_title="FinSight AI - Fraud Detection", layout="centered")

st.title("💳 FinSight AI - Fraud Detection Dashboard")

st.markdown("""
Welcome to the FinSight AI fraud detection tool.

Upload a CSV or manually enter transaction features to test your model.

You can choose to predict using:
- 🚀 FastAPI (real-time API)
- 🧠 Local Model (offline TensorFlow)
""")

# =============================
# SELECT MODE
# =============================
mode = st.radio("Choose prediction mode:", ["🚀 FastAPI", "🧠 Local Model"])

# =============================
# TEST API CONNECTION
# =============================
if mode == "🚀 FastAPI":
    if st.button("🔌 Test API Connection"):
        try:
            res = requests.get(f"{API_URL}/")
            if res.status_code == 200:
                st.success("✅ FastAPI is live: " + res.json()["message"])
            else:
                st.warning("⚠️ FastAPI responded but not OK")
        except Exception as e:
            st.error(f"❌ Could not connect to FastAPI: {str(e)}")

# =============================
# 📂 CSV UPLOAD SECTION
# =============================
st.subheader("📂 Upload Dataset for Bulk Fraud Detection")

st.markdown("""
- File must contain **29 numerical features** per row  
- Columns like `Time` and `Class` will be auto-removed  
- Need a sample? 👉 [Download example CSV](https://raw.githubusercontent.com/Jaykumar-Kotiya/finsight-ai/main/streamlit_app/sample_transactions.csv)
""")

uploaded_file = st.file_uploader("Upload a CSV file with 29 features", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        df.drop(columns=[col for col in ['Time', 'Class'] if col in df.columns], inplace=True)

        if df.shape[1] != EXPECTED_FEATURES:
            st.error(f"❌ Expected 29 features, but found {df.shape[1]}")
        else:
            st.success("✅ File accepted. Running predictions...")

            if mode == "🚀 FastAPI":
                try:
                    res = requests.post(f"{API_URL}/predict_batch", json={"input": df.values.tolist()})
                    if res.status_code == 200:
                        df["Prediction"] = res.json()["predictions"]
                    else:
                        st.error("⚠️ API Error: " + res.text)
                except Exception as e:
                    st.error(f"❌ Could not connect to API: {str(e)}")

            else:
                model = load_local_model()
                preds = model.predict(np.array(df).reshape(df.shape[0], 1, df.shape[1]))
                df["Prediction"] = [int(p[0] > 0.5) for p in preds]

            st.dataframe(df.head(10))

            fig = px.histogram(df, x="Prediction", color="Prediction", barmode="group",
                               title="🧮 Fraud vs Non-Fraud (CSV Upload)",
                               labels={"Prediction": "Fraud Flag"})
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# =============================
# ✍️ MANUAL ENTRY
# =============================
st.subheader("🧮 Manually Enter Transaction Features")

features = []
for i in range(EXPECTED_FEATURES):
    val = st.number_input(f"Feature {i+1}", value=0.0, step=0.1)
    features.append(val)

if "history" not in st.session_state:
    st.session_state.history = []

if st.button("🔍 Predict Fraud (Manual Entry)"):
    try:
        if mode == "🚀 FastAPI":
            res = requests.post(f"{API_URL}/predict", json={"input": features})
            if res.status_code == 200:
                out = res.json()
                confidence = out["confidence"]
                fraud_flag = out["fraud"]
            else:
                st.error("❌ API Error: " + res.text)
                confidence, fraud_flag = None, None
        else:
            model = load_local_model()
            input_array = np.array(features).reshape((1, 1, EXPECTED_FEATURES))
            prediction = model.predict(input_array)
            confidence = float(prediction[0][0])
            fraud_flag = confidence > 0.5

        if confidence is not None:
            st.success(f"✅ Prediction complete — Fraud: {'🛑 YES' if fraud_flag else '✅ NO'}")
            st.markdown(f"**Confidence:** `{confidence:.4f}`")
            st.progress(confidence)

            st.session_state.history.append({
                "input": features,
                "fraud": fraud_flag,
                "confidence": confidence
            })

    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")

# =============================
# 📊 Chart: Manual Prediction History
# =============================
if st.session_state.history:
    fraud_count = sum(1 for h in st.session_state.history if h["fraud"])
    non_fraud_count = len(st.session_state.history) - fraud_count

    st.subheader("📊 Manual Prediction History")
    fig, ax = plt.subplots()
    ax.pie([fraud_count, non_fraud_count],
           labels=["Fraud", "Non-Fraud"],
           colors=["red", "green"],
           autopct="%1.1f%%")
    st.pyplot(fig)
