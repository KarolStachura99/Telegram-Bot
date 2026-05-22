import spacy
import stanza
import requests
from langdetect import detect, LangDetectException

# Słowniki cache na modele, by nie ładować ich przy każdym zapytaniu
_spacy_models = {}
_stanza_pipelines = {}

def get_wikipedia_link(query: str, lang: str) -> str | None:
    """Szuka bytu w Wikipedii za pomocą jej oficjalnego API (NEL/NED)."""
    url = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {"User-Agent": "WSEI_NLP_Bot/1.0 (Student Project)"}
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "srlimit": 1
    }
    try:
        response = requests.get(url, headers=headers, params=params, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data["query"]["search"]:
                title = data["query"]["search"][0]["title"]
                return f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"
    except Exception:
        pass
    return None

def analyze_entities(text: str, method: str = "spacy") -> str:
    """
    Rozpoznaje byty (NER) przy użyciu wybranej metody (spaCy / Stanza)
    oraz linkuje je do bazy wiedzy (NEL).
    """
    # 1. Detekcja języka
    try:
        lang = detect(text)
        if lang not in ['pl', 'en']:
            lang = 'en'
    except LangDetectException:
        lang = 'en'

    result = f" **Wyniki analizy NER & NEL (Metoda: {method.upper()}):**\n\n"

    # --- PODEJŚCIE 1: STANZA ---
    if method.lower() == "stanza":
        if lang not in _stanza_pipelines:
            # Stanza automatycznie pobierze brakujące zasoby przy pierwszym wywołaniu
            stanza.download(lang, processors='tokenize,pos,lemma,ner', logging_level='WARN')
            _stanza_pipelines[lang] = stanza.Pipeline(lang=lang, processors='tokenize,pos,lemma,ner', logging_level='WARN')
        
        nlp = _stanza_pipelines[lang]
        doc = nlp(text)
        
        if not doc.entities:
            return " Stanza przeanalizowała tekst, ale nie wykryła żadnych obiektów nazwanych."

        for ent in doc.entities:
            result += f" **{ent.text}** `[{ent.type}]`\n"
            
            # Pobieramy lemat dla pierwszego słowa z encji (Stanza przechowuje lematy w strukturze słów)
            lemma = ent.words[0].lemma if ent.words else ent.text
            
            # Filtrowanie typów encji dla Stanza (standard: PER, ORG, LOC, MISC)
            # Filtrowanie typów encji dla Stanza (dodane polskie tagi NKJP)
            if ent.type in ["PER", "ORG", "LOC", "GPE", "persName", "orgName", "placeName", "geogName"]:
                wiki_url = get_wikipedia_link(lemma, lang)
                if wiki_url:
                    result += f"    [Wikipedia]({wiki_url})\n"
                else:
                    result += "    *Brak artykułu w bazie*\n"
            result += "\n"

    # --- PODEJŚCIE 2: SPACY ---
    else:
        if lang not in _spacy_models:
            # Używamy pobranego przez Ciebie dużego modelu dla języka polskiego
            model_name = "pl_core_news_lg" if lang == 'pl' else "en_core_web_sm"
            try:
                _spacy_models[lang] = spacy.load(model_name)
            except OSError:
                return f" Błąd: Brak modelu spaCy '{model_name}'. Pobierz go poleceniem terminala."
        
        nlp = _spacy_models[lang]
        doc = nlp(text)

        if not doc.ents:
            return " spaCy przeanalizowało tekst, ale nie wykryło żadnych obiektów nazwanych."

        for ent in doc.ents:
            result += f" **{ent.text}** `[{ent.label_}]`\n"
            
            # Filtrowanie typów encji dla spaCy (dodane polskie tagi NKJP)
            if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "persName", "orgName", "placeName"]:
                wiki_url = get_wikipedia_link(ent.lemma_, lang)
                if wiki_url:
                    result += f"   [Wikipedia]({wiki_url})\n"
                else:
                    result += "   *Brak artykułu w bazie*\n"
            result += "\n"

    return result