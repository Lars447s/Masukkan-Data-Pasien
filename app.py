import streamlit as st
import joblib
import pandas as pd

# 1. Load model yang sudah disimpan
@st.cache_resource
def load_model():
    return joblib.load('logistic_regression_model.pkl')

model = load_model()

st.title("Prediksi Model Logistic Regression")

# 2. Buat form input sesuai dengan fitur yang dibutuhkan model
# Berdasarkan file Anda, fitur yang digunakan meliputi:
# age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
st.header("Masukkan Data Pasien")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50)
    sex = st.selectbox("Sex", [0, 1])
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure", value=120)
    chol = st.number_input("Cholesterol", value=200)

with col2:
    fbs = st.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.selectbox("Resting ECG", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate", value=150)
    exang = st.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.number_input("Oldpeak", value=1.0)
    slope = st.selectbox("Slope", [0, 1, 2])
    ca = st.selectbox("Number of Vessels (ca)", [0, 1, 2, 3])
    thal = st.selectbox("Thal", [0, 1, 2, 3])

# 3. Proses Prediksi
if st.button("Prediksi"):
    # Buat DataFrame dari input user
    input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                thalach, exang, oldpeak, slope, ca, thal]],
                              columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                                       'restecg', 'thalach', 'exang', 'oldpeak', 
                                       'slope', 'ca', 'thal'])
    
    # Karena model adalah pipeline, ia akan otomatis melakukan scaling & encoding
    prediction = model.predict(input_data)
    
    st.success(f"Hasil Prediksi: {prediction[0]}")