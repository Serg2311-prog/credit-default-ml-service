import kagglehub
import shutil
import os

# скачиваем
path = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")

# создаём папку data
os.makedirs("data", exist_ok=True)

# копируем файл в проект
for file in os.listdir(path):
    if file.endswith(".csv"):
        shutil.copy(os.path.join(path, file), "data/UCI_Credit_Card.csv")

print("Dataset downloaded to data/UCI_Credit_Card.csv")
