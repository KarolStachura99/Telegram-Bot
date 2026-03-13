import os
import json

# Ścieżka względem głównego folderu uruchomieniowego (main.py)
DATA_FILE = "data/sentences.json"

def save_sentence_to_json(text: str, text_class: str):
    """Zapisuje nowy rekord do pliku JSON zgodnie z wymogami laboratorium."""
    new_record = {"text": text, "class": text_class}
    
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = [] 
    else:
        os.makedirs("data", exist_ok=True)
        data = []
        
    data.append(new_record)
    
    with open(DATA_FILE, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)