"""Web app Flask: Prediksi Penyakit Jantung.

Menjalankan lokal:
    pip install -r requirements.txt
    python app.py
"""
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

model = joblib.load("model_heart.pkl")
FITUR = joblib.load("fitur_heart.pkl")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], bins=[0, 45, 60, 200], labels=[0, 1, 2]).astype(int)
    d["bp_chol_ratio"] = d["trestbps"] / d["chol"]
    return d


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    try:
        data = request.get_json(force=True)
        row = {k: float(data[k]) for k, _ in FIELDS}
    except (KeyError, ValueError, TypeError):
        return jsonify({"error": "Input tidak valid. Pastikan 13 fitur terisi angka."}), 400

    X = add_features(pd.DataFrame([row]))[FITUR]
    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])
    return jsonify({
        "prediksi": pred,
        "label": "Risiko Tinggi" if pred == 1 else "Risiko Rendah",
        "probabilitas": round(proba, 4),
    })


# (key, label Indonesia) — dipakai validasi input
FIELDS = [
    ("age", "Umur"), ("sex", "Jenis kelamin"), ("cp", "Jenis nyeri dada"),
    ("trestbps", "Tekanan darah"), ("chol", "Kolesterol"), ("fbs", "Gula darah puasa"),
    ("restecg", "EKG istirahat"), ("thalach", "Detak jantung maks"),
    ("exang", "Angina olahraga"), ("oldpeak", "Oldpeak"),
    ("slope", "Slope ST"), ("ca", "Pembuluh utama"), ("thal", "Thalassemia"),
]


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8501)
