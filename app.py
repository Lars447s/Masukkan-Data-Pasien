import streamlit as st
import joblib
import pandas as pd

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Prediksi Kesehatan Jantung",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. KUSTOMISASI CSS DENGAN HTML ---
# Ini untuk membuat tombol lebih menarik dan mengubah sedikit warna dasar
st.markdown("""
    <style>
    /* Styling untuk tombol Prediksi */
    div.stButton > button:first-child {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 24px;
        width: 100%;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #ff3333;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        transform: translateY(-2px);
    }
    /* Styling untuk container hasil */
    .hasil-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# --- 3. MEMUAT MODEL ---
@st.cache_resource
def load_model():
    return joblib.load('logistic_regression_model.pkl')

model = load_model()

# --- 4. SIDEBAR (PANEL SAMPING) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004451.png", width=150) # Ikon ilustrasi medis
    st.title("Tentang Aplikasi")
    st.info(
        "Aplikasi ini menggunakan model **Machine Learning (Logistic Regression)** "
        "untuk memprediksi potensi penyakit jantung berdasarkan data klinis pasien."
    )
    st.markdown("---")
    st.markdown("**Panduan Singkat:**")
    st.markdown("- **0** : Indikator Negatif / Normal")
    st.markdown("- **1, 2, 3** : Indikator Positif / Gejala tertentu")
    st.markdown("---")
    st.caption("Dibuat untuk keperluan akademik/praktikum.")

# --- 5. HEADER UTAMA ---
st.title("🫀 Sistem Prediksi Kesehatan Jantung")
st.markdown("Silakan masukkan data pemeriksaan klinis pasien pada formulir di bawah ini untuk melihat hasil prediksi.")
st.markdown("---")

# --- 6. FORMULIR INPUT DIBAGI MENJADI 3 KOLOM ---
# Membagi ke dalam 3 kolom agar layout lebih rapi dan tidak memanjang ke bawah
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("👤 Data Diri & Gejala Utama")
    age = st.number_input("Usia (Age)", min_value=1, max_value=120, value=50, help="Usia pasien dalam tahun")
    sex = st.selectbox("Jenis Kelamin (Sex)", options=[0, 1], format_func=lambda x: "1 - Pria" if x == 1 else "0 - Wanita")
    cp = st.selectbox("Tipe Nyeri Dada (Chest Pain)", [0, 1, 2, 3], help="0: Asymptomatic, 1: Atypical Angina, 2: Non-anginal, 3: Typical Angina")
    trestbps = st.number_input("Tekanan Darah (Resting BP)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Kolesterol (Cholesterol)", min_value=100, max_value=600, value=200)

with col2:
    st.subheader("🩺 Indikator Klinis")
    fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (FBS)", [0, 1], format_func=lambda x: "1 - Ya (>120)" if x == 1 else "0 - Tidak (<120)")
    restecg = st.selectbox("Hasil EKG (Resting ECG)", [0, 1, 2])
    thalach = st.number_input("Detak Jantung Maks (Max Heart Rate)", min_value=60, max_value=250, value=150)
    exang = st.selectbox("Angina karena Olahraga (Exang)", [0, 1], format_func=lambda x: "1 - Ya" if x == 1 else "0 - Tidak")

with col3:
    st.subheader("🔬 Data Lanjutan")
    oldpeak = st.number_input("Depresi ST (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Kemiringan ST (Slope)", [0, 1, 2])
    ca = st.selectbox("Jumlah Pembuluh Darah (CA)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia (Thal)", [0, 1, 2, 3])

st.markdown("---")

# --- 7. TOMBOL PREDIKSI & HASIL ---
# Menaruh tombol di tengah menggunakan kolom bantu
_, center_col, _ = st.columns([1, 2, 1])

with center_col:
    if st.button("🔍 Lakukan Prediksi", use_container_width=True):
        # Buat DataFrame
        input_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, 
                                    thalach, exang, oldpeak, slope, ca, thal]],
                                  columns=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
                                           'restecg', 'thalach', 'exang', 'oldpeak', 
                                           'slope', 'ca', 'thal'])
        
        # Simulasi loading agar terlihat lebih nyata/profesional
        with st.spinner("Menganalisis data klinis..."):
            prediction = model.predict(input_data)
        
        # Menampilkan hasil dengan kotak peringatan yang cantik
        st.markdown("<br>", unsafe_allow_html=True) # Spasi kosong
        
        if prediction[0] == 0:
            st.success("✅ **HASIL: NEGATIF (0)**")
            st.info("Berdasarkan data yang dimasukkan, pasien diprediksi **TIDAK** memiliki indikasi penyakit jantung. Tetap jaga pola makan dan gaya hidup sehat!")
            st.balloons() # Efek balon jika sehat
        else:
            st.error("🚨 **HASIL: POSITIF (1)**")
            st.warning("Berdasarkan data yang dimasukkan, pasien diprediksi **MEMILIKI** indikasi penyakit jantung. Disarankan untuk segera melakukan pemeriksaan lebih lanjut ke dokter spesialis.")
