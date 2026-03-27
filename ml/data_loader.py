from sklearn.datasets import fetch_20newsgroups

def load_dataset(dataset_name: str) -> tuple:
    """
    Pobiera i przygotowuje zbiór danych do eksperymentu.
    Zwraca krotkę: (lista_tekstów, lista_etykiet, nazwy_klas)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == "20news_group":
        # Pobieramy podzbiór danych, aby nie obciążać pamięci i przyspieszyć trening.
        # Wybieramy tylko 4 kategorie tekstów
        categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
        dataset = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
        
        texts = dataset.data
        labels = dataset.target
        class_names = dataset.target_names
        
        return texts, labels, class_names
        
    else:
        raise ValueError(f"Nieznany zbiór danych: {dataset_name}")
