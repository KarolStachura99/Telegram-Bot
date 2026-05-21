from textblob import TextBlob
import stanza
from transformers import pipeline
import logging

# ==========================================
# ZMIENNE GLOBALNE 
# ==========================================
nlp_stanza = None
transformer_nlp = None

# ==========================================
# WZORZEC LAZY LOADING (Inicjalizacja na żądanie)
# ==========================================
def get_stanza_pipeline():
    global nlp_stanza
    if nlp_stanza is None:
        print("[System AI]: Pierwsze wywołanie. Ładowanie modelu Stanza do pamięci RAM...")
        logging.getLogger('stanza').setLevel(logging.ERROR)
        stanza.download('en', processors='tokenize,sentiment', verbose=False)
        nlp_stanza = stanza.Pipeline(lang='en', processors='tokenize,sentiment', verbose=False)
    return nlp_stanza

def get_transformer_pipeline():
    global transformer_nlp
    if transformer_nlp is None:
        print("[System AI]: Pierwsze wywołanie. Ładowanie modelu Transformer do pamięci RAM...")
        logging.getLogger('transformers').setLevel(logging.ERROR)
        transformer_nlp = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    return transformer_nlp

# ==========================================
# FUNKCJE KLASYFIKUJĄCE
# ==========================================
def rule_based_sentiment(text: str) -> tuple:
    positive_words = ['świetny', 'dobry', 'super', 'polecam', 'uwielbiam', 'wspaniały', 'najlepszy', 'good', 'great', 'awesome']
    negative_words = ['słaby', 'zły', 'fatalny', 'okropny', 'rozczarowany', 'uszkodzony', 'najgorszy', 'bad', 'terrible', 'awful']
    
    text_lower = text.lower()
    pos_count = sum(1 for word in positive_words if word in text_lower)
    neg_count = sum(1 for word in negative_words if word in text_lower)
    
    if pos_count > neg_count:
        return "pozytywny", 1.0
    elif neg_count > pos_count:
        return "negatywny", 1.0
    else:
        return "neutralny", 0.5

def textblob_sentiment(text: str) -> tuple:
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    
    if polarity > 0.1:
        return "pozytywny", polarity
    elif polarity < -0.1:
        return "negatywny", abs(polarity)
    else:
        return "neutralny", 0.5

def stanza_sentiment(text: str) -> tuple:
    # Pobieramy instancję modelu tylko gdy funkcja jest użyta
    nlp = get_stanza_pipeline()
    doc = nlp(text)
    sentiments = [sentence.sentiment for sentence in doc.sentences]
    
    if not sentiments:
        return "neutralny", 1.0
        
    avg_sentiment = sum(sentiments) / len(sentiments)
    
    if avg_sentiment > 1.2:
        return "pozytywny", avg_sentiment / 2.0
    elif avg_sentiment < 0.8:
        # Odwracamy skalę dla negatywnych, by zachować ustandaryzowaną pewność [0, 1]
        return "negatywny", (2.0 - avg_sentiment) / 2.0 
    else:
        return "neutralny", 0.5

def transformer_sentiment(text: str) -> tuple:
    # Pobieramy instancję modelu tylko gdy funkcja jest użyta
    nlp = get_transformer_pipeline()
    result = nlp(text)[0]
    label = result['label'].lower()
    score = result['score']
    
    if label == "positive":
        return "pozytywny", score
    elif label == "negative":
        return "negatywny", score
    else:
        return "neutralny", score

# ==========================================
# GŁÓWNY DYSPOZYTOR
# ==========================================
def analyze_sentiment(method: str, text: str) -> tuple:
    method = method.lower()
    
    if method == "rule":
        return rule_based_sentiment(text)
    elif method == "textblob":
        return textblob_sentiment(text)
    elif method == "stanza":
        return stanza_sentiment(text)
    elif method == "transformer":
        return transformer_sentiment(text)
    else:
        raise ValueError(f"Metoda '{method}' nie jest obsługiwana. Wybierz: rule, textblob, stanza, transformer.")