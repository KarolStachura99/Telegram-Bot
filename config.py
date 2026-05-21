import os

# Podstawowe ustawienia API
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

# Ścieżki do plików i katalogów
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_LAB3_DIR = os.path.join(BASE_DIR, "lab3plots")
DATASET_PATH = os.path.join(BASE_DIR, "sentiment_dataset.csv")

# Parametry uczenia sieci neuronowych
MAX_LEN = 150
EMBEDDING_DIM = 50
BATCH_SIZE = 32
EPOCHS = 7

# Tworzenie głównych folderów przy starcie, by uniknąć FileNotFoundError
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_LAB3_DIR, exist_ok=True)