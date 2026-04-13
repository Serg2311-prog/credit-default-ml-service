# Credit Default ML Service

## Описание
Production-like минималистичный сервис для прогнозирования дефолта по кредитной карте (binary classification) с:

- обучением модели (scikit-learn)
- Flask API (`/health`, `/predict`)
- Docker-упаковкой
- базовыми тестами
- планом A/B тестирования

---

## Структура проекта


.
├── app/
│ ├── init.py
│ ├── api.py
│ └── model_handler.py
├── models/
│ ├── init.py
│ ├── train.py
│ └── artifacts/
│ ├── model.pkl
│ └── metadata.json
├── tests/
│ └── test_api.py
├── docker/
│ └── Dockerfile
├── .dockerignore
├── .gitignore
├── ab_test_plan.md
├── download_data.py
├── requirements.txt
└── README.md


---

## Модель

**Алгоритм:** LogisticRegression  

**Preprocessing:**
- SimpleImputer(strategy="median")
- StandardScaler()

**Артефакты:**
- `models/artifacts/model.pkl`
- `models/artifacts/metadata.json`

---

## Загрузка данных

```bash
python download_data.py
Обучение модели
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python models/train.py

После запуска создаются:

model.pkl
metadata.json (версия и метрики)
Flask API
Эндпоинты
GET /health — проверка готовности сервиса
POST /predict — получение предсказания

Возвращает:

prediction
probability
model_version
Пример запроса
{
  "credit_limit": 6000,
  "age": 34,
  "bill_amount": 2400,
  "payment_amount": 900,
  "late_payments_6m": 2
}
Пример ответа
{
  "prediction": 0,
  "probability": 0.27,
  "model_version": "v1"
}
Запуск API
Linux / Mac
export MODEL_PATH=models/artifacts/model.pkl
flask --app app.api run --host 0.0.0.0 --port 5000
Windows (PowerShell)
$env:MODEL_PATH="models/artifacts/model.pkl"
flask --app app.api run --host 0.0.0.0 --port 5000
Примеры запросов
curl http://localhost:5000/health
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "credit_limit": 6000,
    "age": 34,
    "bill_amount": 2400,
    "payment_amount": 900,
    "late_payments_6m": 2
  }'
Тесты
pytest -q
Docker
Сборка
docker build -f docker/Dockerfile -t credit-default-service:latest .
Запуск
docker run --rm -p 5000:5000 -e MODEL_PATH=models/artifacts/model.pkl credit-default-service:latest
Docker Hub

Docker image: (добавить ссылку после публикации)

Архитектура
Выбор: монолит

Почему:

минимальная сложность для MVP
быстрый запуск и деплой
низкие операционные издержки
достаточно для одного ML use-case
Когда переходить к микросервисам:
рост нагрузки
несколько моделей
разные SLA
Production-концепты
RabbitMQ
асинхронная обработка задач
batch scoring
retraining jobs
Логирование
JSON-логи
request_id, latency, model_version
ELK / Grafana stack
MLflow
трекинг экспериментов
registry моделей
DVC
versioning датасетов
воспроизводимость пайплайна
A/B тестирование

Подробный план: ab_test_plan.md

Бизнес-метрики
expected loss reduction
approval rate at fixed risk
