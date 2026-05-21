import pandas as pd
import os
import config

def load_dataset(dataset_name: str) -> tuple:
    """Ładuje i przygotowuje wskazany zbiór danych."""
    dataset_name = dataset_name.lower()
    
    if dataset_name == "20news_group":
        from sklearn.datasets import fetch_20newsgroups
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
        return dataset.data, dataset.target, dataset.target_names

    elif dataset_name == "imdb":
        from datasets import load_dataset as hf_load
        dataset = hf_load("imdb", split="train[:800]+train[-800:]").shuffle(seed=42)
        return dataset['text'], dataset['label'], ["negatywna", "pozytywna"]
        
    elif dataset_name == "amazon":
        from datasets import load_dataset as hf_load
        dataset = hf_load("amazon_polarity", split="train[:500]+train[-500:]").shuffle(seed=42)
        return dataset['content'], dataset['label'], ["negatywna", "pozytywna"]
        
    elif dataset_name == "custom":
        # Używamy ścieżki bezwzględnej z config.py, żeby bot zawsze trafiał w ten sam plik
        if not os.path.exists(config.DATASET_PATH):
            df = pd.DataFrame(columns=["text", "label"])
            df.to_csv(config.DATASET_PATH, index=False)
            return [], [], ["negatywny", "neutralny", "pozytywny"]
        
        df = pd.read_csv(config.DATASET_PATH)
        if df.empty:
            return [], [], ["negatywny", "neutralny", "pozytywny"]
            
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        return df['text'].tolist(), df['label'].tolist(), ["negatywny", "neutralny", "pozytywny"]
        
    else:
        raise ValueError(f"Nieznany zbiór danych: {dataset_name}")

def add_to_custom_dataset(text: str, label: str):
    """Dopisuje nowy wiersz do naszego własnego zbioru."""
    df = pd.DataFrame([{"text": text, "label": label}])
    if not os.path.exists(config.DATASET_PATH):
        df.to_csv(config.DATASET_PATH, index=False)
    else:
        df.to_csv(config.DATASET_PATH, mode='a', header=False, index=False)