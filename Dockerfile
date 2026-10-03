FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py bot.py start.sh model_heart.pkl fitur_heart.pkl ./
COPY templates/ templates/
COPY static/ static/
RUN chmod +x start.sh

EXPOSE 8501

CMD ["bash", "start.sh"]
