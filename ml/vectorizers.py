import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec
import gensim.downloader as api
import nltk
from nltk.tokenize import word_tokenize

# Upewniamy się, że tokenizer jest dostępny
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def get_document_embedding(text: str, model, vector_size: int) -> np.ndarray:
    """Funkcja pomocnicza do uśredniania wektorów słów w dokumencie dla Word2Vec/GloVe."""
    tokens = word_tokenize(text.lower())
    vectors = [model[word] for word in tokens if word in model]
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

def vectorize_texts(method: str, texts: list) -> np.ndarray:
    """
    Zamienia listę tekstów na macierz cech wybraną metodą.
    Obsługiwane metody: bow, tfidf, word2vec, glove.
    """
    method = method.lower()
    
    if method == "bow":
        # Używamy CountVectorizer z ograniczeniem do 5000 najczęstszych słów
        vectorizer = CountVectorizer(max_features=5000)
        return vectorizer.fit_transform(texts).toarray()
        
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(max_features=5000)
        return vectorizer.fit_transform(texts).toarray()
        
    elif method == "word2vec":
        # Trenujemy własny model Word2Vec na podanym korpusie tekstów
        tokenized_texts = [word_tokenize(text.lower()) for text in texts]
        vector_size = 100
        w2v_model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=2, workers=4)
        
        # Uśredniamy wektory dla każdego dokumentu
        X = np.array([get_document_embedding(text, w2v_model.wv, vector_size) for text in texts])
        return X
        
    elif method == "glove":
        # Pobieramy najlżejszy dostępny model GloVe (50 wymiarów, ok. 66 MB)
        # Przy pierwszym uruchomieniu to pobieranie może chwilę zająć!
        print("Ładowanie modelu GloVe (to może potrwać przy pierwszym uruchomieniu)...")
        glove_model = api.load("glove-wiki-gigaword-50")
        vector_size = 50
        
        X = np.array([get_document_embedding(text, glove_model, vector_size) for text in texts])
        return X
        
    else:
        raise ValueError(f"Nieznana metoda wektoryzacji: {method}")