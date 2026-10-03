#!/bin/bash
# Menjalankan web (gunicorn) dan bot Telegram dalam satu container.
# Container restart otomatis oleh Koyeb jika salah satu proses berhenti.
set -u

gunicorn app:app --bind "0.0.0.0:${PORT}" --workers 1 --threads 4 --timeout 120 &
WEB_PID=$!

if [ -z "${TELEGRAM_BOT_TOKEN:-}" ]; then
  echo "TELEGRAM_BOT_TOKEN kosong: hanya web yang berjalan."
  wait "$WEB_PID"
  exit $?
fi

python bot.py &
BOT_PID=$!

# tunggu sampai salah satu proses berhenti, lalu matikan sisanya
wait -n "$WEB_PID" "$BOT_PID"
echo "Salah satu proses berhenti, me-restart container..."
kill "$WEB_PID" "$BOT_PID" 2>/dev/null
exit 1
