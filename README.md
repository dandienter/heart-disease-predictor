# 🫀 Heart Disease Predictor

Aplikasi prediksi risiko penyakit jantung — **Mini Project End-to-End (Bab 14)**
dari praktikum Machine Learning.

🌐 **Web live:** https://heart-disease-predictor-dandie-46126b49.koyeb.app Model dilatih di
[`praktikumML/bab-14-mini-project-end-to-end`](https://github.com/dandienter/praktikumML/tree/main/bab-14-mini-project-end-to-end)
mengikuti seluruh fase CRISP-DM, lalu di-deploy dalam **dua antarmuka**:

| Antarmuka | File | Cara akses |
|---|---|---|
| 🌐 Web app | `app.py` (Flask) | Browser — **live:** https://heart-disease-predictor-dandie-46126b49.koyeb.app |
| 🤖 Bot Telegram | `bot.py` | Chat Telegram |

Keduanya memakai **model yang sama** (`model_heart.pkl`), jadi hasilnya konsisten.

## 📊 Tentang Model

- **Algoritma:** SVM kernel RBF (`StandardScaler` + `SVC`) — terpilih dari 3 kandidat
  (Logistic Regression, Random Forest, SVM) lewat **GridSearchCV 5-fold**
- **Performa di data uji (60 pasien):**

  | Akurasi | Precision | Recall | F1 | ROC-AUC |
  |---|---|---|---|---|
  | 86,7% | 91,7% | 78,6% | 84,6% | 95,7% |

- **Kenapa model ini?** Pada kasus medis, *false negative* (pasien sakit terdeteksi sehat)
  lebih berbahaya daripada *false positive*. Model dipilih berdasarkan **recall tertinggi**
  lalu F1 — bukan sekadar akurasi tertinggi.
- **Dataset:** Heart Disease UCI Cleveland — 297 baris, 13 fitur klinis + 2 fitur turunan
  (`age_group`, `bp_chol_ratio`).

> ⚠️ Hasil prediksi hanya untuk **tujuan edukasi**, bukan diagnosis medis.

## 📁 Struktur Repo

```text
heart-disease-predictor/
├── app.py              # Web app Flask (UI modern)
├── templates/index.html   # Tampilan web
├── bot.py              # Bot Telegram
├── model_heart.pkl     # Model terlatih (pipeline scaler + SVM)
├── fitur_heart.pkl     # Daftar 15 nama fitur (urutan harus sama)
├── data/heart.csv      # Dataset (dokumentasi)
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Install dependensi

```bash
pip install -r requirements.txt
```

### 2a. Web app (Streamlit)

Buka https://heart-disease-predictor-dandie-46126b49.koyeb.app di browser
(atau jalankan lokal: `python app.py` → `http://localhost:8501`).
Isi 13 data klinis di form, klik **🔍 Prediksi Sekarang** → hasil muncul:

- 🟢 **Risiko RENDAH** — probabilitas, misal `12,4%`
- 🔴 **Risiko TINGGI** — probabilitas, misal `87,9%`

### 2b. Bot Telegram

1. Buat bot lewat [@BotFather](https://t.me/BotFather) → dapat token.
2. Jalankan:

```bash
export TELEGRAM_BOT_TOKEN="<token-bot-kamu>"
python bot.py
```

3. Chat ke bot → `/start` → jawab 13 pertanyaan satu per satu → bot membalas
   hasil prediksi + probabilitasnya. Ketik `/batal` untuk berhenti.

## 🧪 Contoh Hasil

**Input (web / bot):** pria, 63 tahun, nyeri dada typical angina, tekanan darah 145,
kolesterol 233, EKG abnormal, detak jantung maks 150, …

**Output:**

```text
🟢 Risiko RENDAH penyakit jantung
Probabilitas: 29,0%
```

**Input lain** (pria 67 tahun, asymptomatic, tekanan 160, kolesterol 286, …):

```text
🔴 Risiko TINGGI penyakit jantung
Probabilitas: 77,1%
Disarankan konsultasi ke dokter untuk pemeriksaan lanjutan.
```

## 🔁 Melatih Ulang Model

Model dilatih dari notebook Bab 14. Untuk melatih ulang / eksperimen:

1. Buka `bab-14-mini-project-end-to-end/praktikum-bab-14.ipynb` di repo `praktikumML`
   (atau via Google Colab).
2. Jalankan semua cell — model terbaik otomatis tersimpan sebagai `model_heart.pkl`.
3. Salin `model_heart.pkl` + `fitur_heart.pkl` yang baru ke repo ini.

## 👤 Author

**Ahmad Dandi Subhani** — 202343500126 — Universitas Indraprasta PGRI
