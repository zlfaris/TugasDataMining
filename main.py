import streamlit as st
import joblib
import numpy as np
import pandas as pd

st.set_page_config(
    page_title="Prediksi Penyakit Jantung",
    page_icon="❤️",
    layout="centered"
)

@st.cache_resource
def load_models():
    try:
        model_bnb = joblib.load("model_bernoulli_nb.pkl")
        model_svm = joblib.load("model_linear_svm.pkl")
        model_ensemble = joblib.load("model_ensemble_voting.pkl")
        return model_bnb, model_svm, model_ensemble
    except:
        return None, None, None

model_bnb, model_svm, model_ensemble = load_models()
models_loaded = all([model_bnb, model_svm, model_ensemble])

st.title("❤️ Prediksi Risiko Penyakit Jantung")
st.markdown("### Ensemble Model (BernoulliNB + SVM)")

if not models_loaded:
    st.error("⚠️ Model tidak ditemukan. Pastikan file `.pkl` sudah ada.")
    st.stop()

st.subheader("🧍‍♂️ Masukkan Data Pasien")

col1, col2 = st.columns(2)

with col1:
    Age = st.number_input("Age", 1, 120, 40)
    RestingBP = st.number_input("RestingBP", 0, 250, 120)
    Cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    FastingBS = st.selectbox("FastingBS (Fasting Blood Sugar > 120 mg/dl?)", [0, 1])

with col2:
    MaxHR = st.number_input("MaxHR", 50, 250, 150)
    Oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
    Sex = st.selectbox("Sex", ["M", "F"])
    ExerciseAngina = st.selectbox("Exercise Angina", ["N", "Y"])

ChestPainType = st.selectbox(
    "Chest Pain Type",
    ["TA", "ATA", "NAP", "ASY"]
)

RestingECG = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

ST_Slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

input_data = pd.DataFrame([{
    "Age": Age,
    "RestingBP": RestingBP,
    "Cholesterol": Cholesterol,
    "FastingBS": FastingBS,
    "MaxHR": MaxHR,
    "Oldpeak": Oldpeak,
    "Sex": Sex,
    "ChestPainType": ChestPainType,
    "RestingECG": RestingECG,
    "ExerciseAngina": ExerciseAngina,
    "ST_Slope": ST_Slope
}])

if st.button("🔍 Prediksi", type="primary"):

    with st.spinner("Menghitung risiko..."):

        # Prediksi semua model
        pred_bnb = model_bnb.predict(input_data)[0]
        pred_svm = model_svm.predict(input_data)[0]
        pred_ensemble = model_ensemble.predict(input_data)[0]

        prob_bnb = model_bnb.predict_proba(input_data)[0]
        prob_svm = model_svm.predict_proba(input_data)[0]
        prob_ensemble = model_ensemble.predict_proba(input_data)[0]

        st.subheader("🎯 Hasil Prediksi (Ensemble)")

        if pred_ensemble == 1:
            st.error("### ⚠️ Risiko Tinggi Penyakit Jantung")
        else:
            st.success("### ✅ Risiko Rendah Penyakit Jantung")

        st.info(f"Probabilitas (Risiko Tinggi): **{prob_ensemble[1]*100:.1f}%**")

        st.subheader("📌 Perbandingan Model")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("BernoulliNB",
                      "High" if pred_bnb == 1 else "Low",
                      f"{prob_bnb[1]*100:.1f}%")

        with colB:
            st.metric("Linear SVM",
                      "High" if pred_svm == 1 else "Low",
                      f"{prob_svm[1]*100:.1f}%")

        with colC:
            st.metric("Ensemble",
                      "High" if pred_ensemble == 1 else "Low",
                      f"{prob_ensemble[1]*100:.1f}%")

