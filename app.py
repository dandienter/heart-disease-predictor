"""Aplikasi web Streamlit: Prediksi Penyakit Jantung.

Menjalankan:
    streamlit run app.py
"""
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prediksi Penyakit Jantung", page_icon="🫀", layout="centered")

@st.cache_resource
def load_assets():
    model = joblib.load("model_heart.pkl")
    fitur = joblib.load("fitur_heart.pkl")
    return model, fitur

model, FITUR = load_assets()

def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], bins=[0, 45, 60, 200], labels=[0, 1, 2]).astype(int)
    d["bp_chol_ratio"] = d["trestbps"] / d["chol"]
    return d

st.title("🫀 Prediksi Risiko Penyakit Jantung")
st.write(
    "Masukkan 13 data klinis pasien di bawah. Model Machine Learning "
    "(SVM-RBF, hasil tuning GridSearchCV di Bab 14 praktikumML) akan memprediksi "
    "tingkat risiko penyakit jantung."
)
st.divider()

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Umur (tahun)", 20, 90, 50)
        sex = st.selectbox("Jenis kelamin", [("Laki-laki", 1), ("Perempuan", 0)],
                           format_func=lambda x: x[0])[1]
        cp = st.selectbox("Jenis nyeri dada",
                          [("Typical angina", 0), ("Atypical angina", 1),
                           ("Non-anginal", 2), ("Asymptomatic", 3)],
                          format_func=lambda x: x[0])[1]
        trestbps = st.slider("Tekanan darah istirahat (mm Hg)", 80, 220, 120)
        chol = st.slider("Kolesterol (mg/dl)", 100, 600, 240)
        fbs = st.selectbox("Gula darah puasa > 120 mg/dl",
                           [("Tidak", 0), ("Ya", 1)], format_func=lambda x: x[0])[1]
        restecg = st.selectbox("Hasil EKG istirahat",
                               [("Normal", 0), ("ST-T abnormal", 1),
                                ("Hipertrofi ventrikel", 2)],
                               format_func=lambda x: x[0])[1]
    with col2:
        thalach = st.slider("Detak jantung maksimum", 70, 220, 150)
        exang = st.selectbox("Angina saat olahraga",
                             [("Tidak", 0), ("Ya", 1)], format_func=lambda x: x[0])[1]
        oldpeak = st.slider("ST depression (oldpeak)", 0.0, 7.0, 1.0, step=0.1)
        slope = st.selectbox("Kemiringan segmen ST",
                             [("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)],
                             format_func=lambda x: x[0])[1]
        ca = st.selectbox("Jumlah pembuluh utama (fluoroskopi)", [0, 1, 2, 3])
        thal = st.selectbox("Thalassemia",
                            [("Normal", 3), ("Fixed defect", 6),
                             ("Reversible defect", 7)],
                            format_func=lambda x: x[0])[1]

    submitted = st.form_submit_button("🔍 Prediksi", use_container_width=True)

if submitted:
    row = {"age": age, "sex": sex, "cp": cp, "trestbps": trestbps,
           "chol": chol, "fbs": fbs, "restecg": restecg, "thalach": thalach,
           "exang": exang, "oldpeak": oldpeak, "slope": slope, "ca": ca,
           "thal": thal}
    X = add_features(pd.DataFrame([row]))[FITUR]
    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    st.divider()
    if pred == 1:
        st.error(f"🔴 **Risiko TINGGI penyakit jantung** — probabilitas {proba:.1%}")
        st.write("Disarankan konsultasi ke dokter untuk pemeriksaan lanjutan.")
    else:
        st.success(f"🟢 **Risiko RENDAH penyakit jantung** — probabilitas {proba:.1%}")
        st.write("Tetap jaga pola hidup sehat dan cek kesehatan berkala.")

    st.caption(
        "⚠️ Hasil ini adalah prediksi model Machine Learning untuk tujuan edukasi, "
        "bukan diagnosis medis. Konsultasikan ke tenaga medis profesional."
    )

with st.expander("ℹ️ Tentang model"):
    st.write(
        "- **Algoritma:** SVM kernel RBF (StandardScaler + SVC), terpilih dari 3 kandidat "
        "(Logistic Regression, Random Forest, SVM) lewat GridSearchCV 5-fold.\n"
        "- **Performa di data uji:** akurasi 86,7% · recall 78,6% · F1 84,6% · ROC-AUC 95,7%.\n"
        "- **Dataset:** Heart Disease UCI Cleveland (297 baris setelah cleaning).\n"
        "- Detail lengkap: Bab 14 di repo `praktikumML`."
    )
