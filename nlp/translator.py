from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect, LangDetectException

# Zmieniamy strukturę cache'u, żeby trzymał i model, i tokenizator osobno
_models_cache = {}

def translate_text_with_transformers(text: str) -> str:
    """
    Wykrywa język tekstu i tłumaczy go przy użyciu lokalnych modeli Hugging Face.
    Omija funkcję pipeline na rzecz bezpośredniego wywołania modelu.
    """
    # 1. Detekcja języka tekstu
    try:
        lang = detect(text)
    except LangDetectException:
        return " Błąd: Nie udało się rozpoznać języka. Podaj dłuższy tekst."

    # 2. Wybór odpowiedniego modelu kierunkowego
    if lang == 'pl':
        model_name = "Helsinki-NLP/opus-mt-pl-en"
        direction = "🇵🇱 ➡️ 🇬🇧"
    elif lang == 'en':
        model_name = "Helsinki-NLP/opus-mt-en-pl"
        direction = "🇬🇧 ➡️ 🇵🇱"
    else:
        return f" Wykryto język: '{lang}'. Obecnie obsługuję tylko tłumaczenie między Polskim a Angielskim."

    # 3. Pobranie i załadowanie modelu oraz tokenizatora
    if model_name not in _models_cache:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        _models_cache[model_name] = {"tokenizer": tokenizer, "model": model}

    tokenizer = _models_cache[model_name]["tokenizer"]
    model = _models_cache[model_name]["model"]

    # 4. Ręczne przygotowanie danych (Tokenizacja)
    # Zamieniamy tekst na tensory zrozumiałe dla sieci neuronowej
    inputs = tokenizer(text, return_tensors="pt", padding=True)

    # 5. Wykonanie tłumaczenia (Generacja)
    outputs = model.generate(**inputs)

    # 6. Dekodowanie tensorów z powrotem na tekst
    translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return f" **Tłumaczenie ({direction}):**\n\n{translated_text}"