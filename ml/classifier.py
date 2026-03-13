import json
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_FILE = "data/sentences.json"

def train_and_predict(new_text: str) -> str:
    """Trenuje model na danych z JSON i zwraca przewidzianą klasę."""
    # 1. Bezpieczne wczytanie danych
    if not os.path.exists(DATA_FILE):
        return "BŁĄD: Brak pliku z danymi. Najpierw dodaj teksty przez /task."
        
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            return "BŁĄD: Plik JSON jest uszkodzony."
            
    if len(data) < 2:
        return "BŁĄD: Za mało danych do treningu. Dodaj więcej przykładów."

    texts = [item["text"] for item in data]
    raw_labels = [item["class"] for item in data]

    # 2. Mapowanie klas na wartości liczbowe
    label_map = {"pozytywny": 1, "neutralny": 0, "negatywny": -1}
    # Jeśli wystąpi inna klasa, traktujemy ją jako neutralną (0)
    labels = [label_map.get(label.lower(), 0) for label in raw_labels]

    # 3. Zbudowanie i wytrenowanie klasyfikatora
    model = Pipeline([
        ("vectorizer", CountVectorizer()),
        ("classifier", LogisticRegression(max_iter=1000))
    ])
    
    try:
        model.fit(texts, labels)
        prediction = model.predict([new_text])[0]
        
        # 4. Mapowanie odwrotne (z liczby na tekst)
        reverse_map = {1: "pozytywny", 0: "neutralny", -1: "negatywny"}
        return reverse_map.get(prediction, "nieznany")
    except Exception as e:
        return f"Błąd trenowania modelu: {e}"