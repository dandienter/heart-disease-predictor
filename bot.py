"""Bot Telegram: Prediksi Penyakit Jantung.

Bot menanyakan 13 fitur klinis satu per satu lewat chat, lalu menjawab
hasil prediksi dari model yang sama dengan aplikasi web.

Menjalankan:
    export TELEGRAM_BOT_TOKEN="<token dari @BotFather>"
    python bot.py
"""
import logging
import os

import joblib
import pandas as pd
from telegram import Update
from telegram.ext import (Application, CommandHandler, ContextTypes,
                          ConversationHandler, MessageHandler, filters)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("model_heart.pkl")
FITUR = joblib.load("fitur_heart.pkl")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], bins=[0, 45, 60, 200], labels=[0, 1, 2]).astype(int)
    d["bp_chol_ratio"] = d["trestbps"] / d["chol"]
    return d


# (key, pertanyaan, parser) — parser mengubah teks user jadi angka
PERTANYAAN = [
    ("age", "[1/13] Umur pasien (tahun)?\nContoh: 55", float),
    ("sex", "[2/13] Jenis kelamin? (1 = laki-laki, 0 = perempuan)", float),
    ("cp", "[3/13] Jenis nyeri dada? (1=typical angina, 2=atypical angina, 3=non-anginal, 4=asymptomatic)", float),
    ("trestbps", "[4/13] Tekanan darah istirahat (mm Hg)?\nContoh: 130", float),
    ("chol", "[5/13] Kolesterol (mg/dl)?\nContoh: 250", float),
    ("fbs", "[6/13] Gula darah puasa > 120 mg/dl? (1 = ya, 0 = tidak)", float),
    ("restecg", "[7/13] Hasil EKG istirahat? (0=normal, 1=ST-T abnormal, 2=hipertrofi ventrikel)", float),
    ("thalach", "[8/13] Detak jantung maksimum yang tercapai?\nContoh: 150", float),
    ("exang", "[9/13] Angina saat olahraga? (1 = ya, 0 = tidak)", float),
    ("oldpeak", "[10/13] ST depression (oldpeak)?\nContoh: 1.4", float),
    ("slope", "[11/13] Kemiringan segmen ST? (1=upsloping, 2=flat, 3=downsloping)", float),
    ("ca", "[12/13] Jumlah pembuluh utama terlihat (fluoroskopi)? (0-3)", float),
    ("thal", "[13/13] Thalassemia? (3=normal, 6=fixed defect, 7=reversible defect)", float),
]

TANYA = 0


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text(
        "*Prediksi Risiko Penyakit Jantung*\n\n"
        "Saya akan menanyakan 13 data klinis, lalu memprediksi tingkat risikonya "
        "dengan model Machine Learning.\n\n"
        "Ketik /batal kapan saja untuk berhenti.",
        parse_mode="Markdown",
    )
    context.user_data["idx"] = 0
    await update.message.reply_text(PERTANYAAN[0][1])
    return TANYA


async def terima(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    idx = context.user_data["idx"]
    key, _, parser = PERTANYAAN[idx]
    try:
        context.user_data[key] = parser(update.message.text.strip().replace(",", "."))
    except ValueError:
        await update.message.reply_text("Masukkan angka yang valid ya. Coba lagi:")
        return TANYA

    idx += 1
    if idx < len(PERTANYAAN):
        context.user_data["idx"] = idx
        await update.message.reply_text(PERTANYAAN[idx][1])
        return TANYA

    # semua terjawab -> prediksi
    row = {k: context.user_data[k] for k, _, _ in PERTANYAAN}
    X = add_features(pd.DataFrame([row]))[FITUR]
    proba = float(model.predict_proba(X)[0][1])
    pred = int(model.predict(X)[0])

    if pred == 1:
        hasil = (f"*Risiko TINGGI penyakit jantung*\n"
                 f"Probabilitas: {proba:.1%}\n\n"
                 "Disarankan konsultasi ke dokter untuk pemeriksaan lanjutan.")
    else:
        hasil = (f"*Risiko RENDAH penyakit jantung*\n"
                 f"Probabilitas: {proba:.1%}\n\n"
                 "Tetap jaga pola hidup sehat dan cek kesehatan berkala.")

    hasil += ("\n\n_Ini prediksi model ML untuk edukasi, bukan diagnosis medis._\n"
              "Ketik /start untuk prediksi baru.")
    await update.message.reply_text(hasil, parse_mode="Markdown")
    return ConversationHandler.END


async def batal(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    await update.message.reply_text("Dibatalkan. Ketik /start untuk mulai lagi.")
    return ConversationHandler.END


def main() -> None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        raise SystemExit("Set TELEGRAM_BOT_TOKEN dulu. Contoh:\n"
                         "  export TELEGRAM_BOT_TOKEN=\"123:ABC\"")
    app = Application.builder().token(token).build()
    conv = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={TANYA: [MessageHandler(filters.TEXT & ~filters.COMMAND, terima)]},
        fallbacks=[CommandHandler("batal", batal)],
    )
    app.add_handler(conv)
    logger.info("Bot berjalan...")
    app.run_polling()


if __name__ == "__main__":
    main()
