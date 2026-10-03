"""Bot Telegram: Prediksi Penyakit Jantung.

Semua pertanyaan dijawab cukup dengan menekan tombol (inline keyboard).
Untuk angka seperti umur, pengguna juga bisa mengetik nilai sendiri
jika tidak ada di tombol.

Menjalankan:
    export TELEGRAM_BOT_TOKEN="<token dari @BotFather>"
    python bot.py
"""
import logging
import os

import joblib
import pandas as pd
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import (Application, CallbackQueryHandler, CommandHandler,
                          ContextTypes, ConversationHandler, MessageHandler,
                          filters)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model = joblib.load("model_heart.pkl")
FITUR = joblib.load("fitur_heart.pkl")


def add_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_group"] = pd.cut(d["age"], bins=[0, 45, 60, 200], labels=[0, 1, 2]).astype(int)
    d["bp_chol_ratio"] = d["trestbps"] / d["chol"]
    return d


# (key, teks pertanyaan, [(label tombol, nilai), ...])
SOAL = [
    ("age", "Umur pasien (tahun)?\nPilih tombol atau ketik angka sendiri.",
     [("30", 30), ("40", 40), ("50", 50), ("55", 55), ("60", 60), ("65", 65), ("70", 70)]),
    ("sex", "Jenis kelamin?",
     [("Laki-laki", 1), ("Perempuan", 0)]),
    ("cp", "Jenis nyeri dada?",
     [("Tipikal", 1), ("Atipikal", 2), ("Non-angina", 3), ("Asimtomatik", 4)]),
    ("trestbps", "Tekanan darah istirahat (mmHg)?\nPilih tombol atau ketik angka sendiri.",
     [("110", 110), ("120", 120), ("130", 130), ("140", 140), ("150", 150), ("160", 160)]),
    ("chol", "Kolesterol (mg/dl)?\nPilih tombol atau ketik angka sendiri.",
     [("180", 180), ("200", 200), ("220", 220), ("250", 250), ("280", 280), ("300", 300)]),
    ("fbs", "Gula darah puasa > 120 mg/dl?",
     [("Ya", 1), ("Tidak", 0)]),
    ("restecg", "Hasil EKG istirahat?",
     [("Normal", 0), ("ST-T abnormal", 1), ("Hipertrofi ventrikel", 2)]),
    ("thalach", "Detak jantung maksimum yang tercapai?\nPilih tombol atau ketik angka sendiri.",
     [("120", 120), ("140", 140), ("150", 150), ("160", 160), ("180", 180)]),
    ("exang", "Nyeri dada saat olahraga?",
     [("Ya", 1), ("Tidak", 0)]),
    ("oldpeak", "Depresi segmen ST (oldpeak)?\nPilih tombol atau ketik angka sendiri (contoh: 1,4).",
     [("0", 0), ("0,5", 0.5), ("1,0", 1.0), ("1,5", 1.5), ("2,0", 2.0), ("3,0", 3.0)]),
    ("slope", "Kemiringan segmen ST?",
     [("Menanjak", 1), ("Datar", 2), ("Menurun", 3)]),
    ("ca", "Jumlah pembuluh utama terlihat (fluoroskopi)?",
     [("0", 0), ("1", 1), ("2", 2), ("3", 3)]),
    ("thal", "Thalassemia?",
     [("Normal", 3), ("Cacat tetap", 6), ("Cacat reversibel", 7)]),
]

TANYA = 0

# soal numerik boleh juga dijawab dengan mengetik angka sendiri
NUMERIK = {"age", "trestbps", "chol", "thalach", "oldpeak"}
BATAS = {
    "age": (1, 120),
    "trestbps": (80, 250),
    "chol": (100, 600),
    "thalach": (60, 250),
    "oldpeak": (0, 10),
}


def keyboard_soal(idx: int) -> InlineKeyboardMarkup:
    _, _, opsi = SOAL[idx]
    baris, tmp = [], []
    for label, nilai in opsi:
        tmp.append(InlineKeyboardButton(label, callback_data=f"soal:{idx}:{nilai}"))
        if len(tmp) == 3:
            baris.append(tmp)
            tmp = []
    if tmp:
        baris.append(tmp)
    return InlineKeyboardMarkup(baris)


def teks_soal(idx: int) -> str:
    _, tanya, _ = SOAL[idx]
    return f"[{idx + 1}/{len(SOAL)}] {tanya}"


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    context.user_data.clear()
    context.user_data["idx"] = 0
    await update.message.reply_text(
        "*Prediksi Risiko Penyakit Jantung*\n\n"
        "Jawab 13 pertanyaan berikut cukup dengan menekan tombol. "
        "Hasil prediksi dihitung dengan model Machine Learning.\n\n"
        "Ketik /batal kapan saja untuk berhenti.",
        parse_mode="Markdown",
    )
    await update.message.reply_text(teks_soal(0), reply_markup=keyboard_soal(0))
    return TANYA


def teks_hasil(user_data: dict) -> str:
    row = {k: user_data[k] for k, _, _ in SOAL}
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
    return hasil


async def jawab(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    try:
        _, idx_s, nilai_s = query.data.split(":")
        idx = int(idx_s)
        nilai = float(nilai_s)
    except (ValueError, AttributeError):
        return TANYA

    # abaikan tombol dari soal yang sudah lewat
    if idx != context.user_data.get("idx", -1):
        return TANYA

    key = SOAL[idx][0]
    context.user_data[key] = nilai
    idx += 1
    context.user_data["idx"] = idx

    if idx < len(SOAL):
        await query.edit_message_text(teks_soal(idx), reply_markup=keyboard_soal(idx))
        return TANYA

    await query.edit_message_text(teks_hasil(context.user_data), parse_mode="Markdown")
    return ConversationHandler.END


async def terima_teks(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Jawaban ketikan manual, hanya untuk soal numerik."""
    idx = context.user_data.get("idx", -1)
    if idx < 0 or idx >= len(SOAL):
        return TANYA

    key = SOAL[idx][0]
    if key not in NUMERIK:
        await update.message.reply_text(
            "Untuk pertanyaan ini pilih salah satu tombol di atas ya.")
        return TANYA

    try:
        nilai = float(update.message.text.strip().replace(",", "."))
    except ValueError:
        await update.message.reply_text("Masukkan angka yang valid, contoh: 45")
        return TANYA

    lo, hi = BATAS[key]
    if not (lo <= nilai <= hi):
        await update.message.reply_text(
            f"Nilainya di luar rentang wajar ({lo}-{hi}). Coba lagi.")
        return TANYA

    context.user_data[key] = nilai
    idx += 1
    context.user_data["idx"] = idx

    if idx < len(SOAL):
        await update.message.reply_text(teks_soal(idx), reply_markup=keyboard_soal(idx))
        return TANYA

    await update.message.reply_text(teks_hasil(context.user_data), parse_mode="Markdown")
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
        states={TANYA: [
            CallbackQueryHandler(jawab, pattern=r"^soal:"),
            MessageHandler(filters.TEXT & ~filters.COMMAND, terima_teks),
        ]},
        fallbacks=[CommandHandler("batal", batal)],
    )
    app.add_handler(conv)
    logger.info("Bot berjalan...")
    app.run_polling()


if __name__ == "__main__":
    main()
