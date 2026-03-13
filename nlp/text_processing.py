import nltk
import os
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Bezpieczne pobieranie modelu tokenizatora przy pierwszym uruchomieniu
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

def tokenize_text(text: str) -> list:
    #Dzieli wprowadzony tekst na listę tokenów.
    return nltk.word_tokenize(text)

# Wczytywanie stop words do pamięci podczas uruchamiania bota
def load_stopwords() -> set:
    """Wczytuje polskie stop words z pliku tekstowego do zbioru (set)."""
    stopwords = set()
    filepath = "data/stopwords_pl.txt"
    
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as file:
            for line in file:
                word = line.strip().lower() # Usuwamy białe znaki i zamieniamy na małe litery
                if word:
                    stopwords.add(word)
    else:
        print(f"OSTRZEŻENIE: Nie znaleziono pliku {filepath}. Stop words nie będą usuwane.")
        
    return stopwords

# Inicjalizujemy zbiór stop słów jako zmienną globalną dla tego modułu
POLISH_STOPWORDS = load_stopwords()

def remove_stopwords_from_tokens(tokens: list) -> list:
    """
    Przyjmuje listę tokenów i zwraca nową listę, 
    pozbawioną słów znajdujących się w słowniku stop words.
    """
    # Zostawiamy tylko te słowa, których (po zamianie na małe litery) NIE MA w naszym zbiorze
    return [word for word in tokens if word.lower() not in POLISH_STOPWORDS]

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Inicjalizujemy narzędzia NLTK
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def stem_tokens(tokens: list) -> list:
    """Stemming - brutalne obcinanie końcówek wyrazów."""
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens: list) -> list:
    """Lematyzacja - sprowadzanie do formy słownikowej."""
    return [lemmatizer.lemmatize(word) for word in tokens]


def get_bag_of_words(text: str) -> str:
    """Tworzy reprezentację Bag of Words dla pojedynczego tekstu."""
    vectorizer = CountVectorizer()
    try:
        # Transformujemy tekst na wektor
        X = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        array = X.toarray()[0]
        # Tworzymy czytelny słownik: słowo -> liczba wystąpień
        bow_dict = dict(zip(features, array))
        return str(bow_dict)
    except ValueError:
        return "Tekst jest za krótki lub pusty, aby stworzyć Bag of Words."

def get_tfidf(text: str) -> str:
    """Tworzy reprezentację TF-IDF dla pojedynczego tekstu."""
    vectorizer = TfidfVectorizer()
    try:
        X = vectorizer.fit_transform([text])
        features = vectorizer.get_feature_names_out()
        array = X.toarray()[0]
        # Tworzymy czytelny słownik: słowo -> waga TF-IDF (zaokrąglona do 2 miejsc)
        tfidf_dict = {feat: round(val, 2) for feat, val in zip(features, array)}
        return str(tfidf_dict)
    except ValueError:
        return "Tekst jest za krótki lub pusty, aby stworzyć TF-IDF."