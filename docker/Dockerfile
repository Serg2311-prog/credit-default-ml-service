FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /service

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY models ./models

EXPOSE 5000

ENV MODEL_PATH=models/artifacts/model.pkl

CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app.api:app"]

