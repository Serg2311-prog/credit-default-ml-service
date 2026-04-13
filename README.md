# Credit Default ML Service

## Описание

Production-like минималистичный сервис для прогнозирования дефолта по кредитной карте  
(binary classification) с:
- обучением модели (`scikit-learn`)
- Flask API (`/health`, `/predict`)
- Docker-упаковкой
- базовыми тестами
- планом A/B тестирования

## 1) Структура проекта

```text
.
├── app/
│   ├── __init__.py
│   ├── api.py
│   └── model_handler.py
├── models/
│   ├── __init__.py
│   ├── train.py
│   └── artifacts/
│       ├── model.pkl
│       └── metadata.json
├── tests/
│   └── test_api.py
├── docker/
│   └── Dockerfile
├── .dockerignore
├── .gitignore
├── ab_test_plan.md
├── requirements.txt
└── README.md
```

## 2) Модель

- Алгоритм: `LogisticRegression`
- Минимальный preprocessing:
  - `SimpleImputer(strategy="median")`
  - `StandardScaler()`
- Сохранение артефактов: `joblib` (`models/artifacts/model.pkl`)
- Скрипт обучения: `models/train.py`

## 3) Загрузка данных

```bash
python download_data.py
python models/train.py
```

## 4) Обучение модели

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python models/train.py
```

После запуска создаются:
- `models/artifacts/model.pkl`
- `models/artifacts/metadata.json` (версия и метрики)

## 5) Flask API

### Эндпоинты

- `GET /health`
  - Проверка готовности сервиса и загрузки модели
- `POST /predict`
  - Принимает JSON c признаками и возвращает:
    - `prediction` (0/1)
    - `probability` (вероятность класса 1)
    - `model_version`

### Пример JSON для /predict

```json
{
  "credit_limit": 6000,
  "age": 34,
  "bill_amount": 2400,
  "payment_amount": 900,
  "late_payments_6m": 2
}
```

### Пример ответа API (POST /predict)

```json
{
  "prediction": 0,
  "probability": 0.27,
  "model_version": "v1"
}
```

### Локальный запуск API

```bash
export MODEL_PATH=models/artifacts/model.pkl
flask --app app.api run --host 0.0.0.0 --port 5000
```

### Запуск на Windows (PowerShell)

```powershell
$env:MODEL_PATH="models/artifacts/model.pkl"
flask --app app.api run --host 0.0.0.0 --port 5000
```

### Примеры curl

```bash
curl -X GET http://localhost:5000/health
```

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_limit": 6000,
    "age": 34,
    "bill_amount": 2400,
    "payment_amount": 900,
    "late_payments_6m": 2
  }'
```

## 6) Запуск тестов

```bash
pytest -q
```

## 7) Docker

Dockerfile находится в `docker/Dockerfile`:
- base image: `python:3.12-slim`
- порт: `5000`
- запуск: `gunicorn app.api:app`

### Сборка образа

```bash
docker build -f docker/Dockerfile -t credit-default-service:latest .
```

### Запуск контейнера

```bash
docker run --rm -p 5000:5000 credit-default-service:latest
```

## 8) Docker Hub

Docker image: <добавить ссылку после публикации>

## 9) Архитектурное решение: монолит vs микросервисы

### Текущий выбор: **монолит**

Почему:
- минимальная сложность для MVP
- быстрее запуск и деплой
- меньше операционных накладных расходов
- достаточно для одного ML use-case

### Когда переходить к микросервисам

Имеет смысл выделять сервисы отдельно (feature service, model serving, experiment tracking), когда:
- сильно растет нагрузка и требования к независимому масштабированию
- появляется много моделей/команд
- есть разные SLA по частям системы

## 10) Production-концепты (следующий шаг)

### RabbitMQ
- Асинхронная обработка задач: batch scoring, retraining jobs, уведомления.
- Снижение нагрузки на API через очередь заданий.

### Логирование
- Структурированные JSON-логи (request_id, latency, model_version, status_code).
- Централизация через ELK/Opensearch/Grafana stack.

### MLflow
- Трекинг экспериментов: параметры, метрики, артефакты, registry моделей.
- Контролируемый выпуск новых версий модели.

### DVC
- Версионирование датасетов и ML артефактов.
- Репродуцируемый training pipeline в CI/CD.

## 11) A/B тестирование

Подробный план в файле: **`ab_test_plan.md`**.

## 12) Бизнес-метрики

- ожидаемое снижение потерь от дефолтов (expected loss reduction)
- доля одобренных заявок при фиксированном уровне риска (approval rate at fixed risk)
