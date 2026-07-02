# Telegram Bot - NLP & Text Classification

Zaawansowany bot na platformę Telegram, zbudowany w języku Python, służący do przetwarzania języka naturalnego (NLP) oraz przeprowadzania eksperymentów z zakresu uczenia maszynowego (Machine Learning), głębokich sieci neuronowych i systemów agentowych. Projekt realizowany w ramach sześciu etapów laboratoryjnych.

---

## Etap 1: Laboratorium 1 - Przetwarzanie tekstu i podstawowa klasyfikacja

W pierwszym etapie zaimplementowano podstawowy rurociąg (pipeline) NLP, pozwalający na analizę i klasyfikację pojedynczych wiadomości tekstowych.

### Główne funkcjonalności:
* **Przetwarzanie tekstu (Preprocessing):** Tokenizacja, usuwanie stop words, stemming oraz lematyzacja.
* **Wektoryzacja:** Zamiana tekstu na reprezentacje liczbowe za pomocą Bag of Words (BoW) oraz TF-IDF.
* **Wizualizacja:** Generowanie chmur słów i histogramów długości tokenów dla podanych zdań (katalog `plots/`).
* **Baza danych:** Zapisywanie przetworzonych zdań i ich klas do pliku `data/sentences.json`.

### Dostępne komendy:
* `/start` - Powitanie i inicjalizacja bota.
* `/task <zadanie> "tekst" "klasa"` - Wykonanie pojedynczego zadania NLP.
* `/full_pipeline "tekst" "klasa"` - Pełny proces przetwarzania tekstu.
* `/classifier "tekst"` - Szybka klasyfikacja wiadomości.
* `/stats` - Generowanie statystyk i wykresów dla całego zbioru.

---

## Etap 2: Laboratorium 2 - Eksperymenty klasyfikacji dla całych zbiorów danych

Moduł badawczy do przeprowadzania zautomatyzowanych eksperymentów na dużych zbiorach danych.

### Główne funkcjonalności:
* **Obsługa datasetów:** Zautomatyzowane pobieranie korpusów (m.in. `20news_group`).
* **Zaawansowane embeddingi:** TF-IDF, BoW, Word2Vec oraz pretrenowany GloVe.
* **Modele klasyfikacji:** Multinomial Naive Bayes (nb), Random Forest (rf), MLPClassifier (mlp), Logistic Regression (logreg).
* **Strojenie hiperparametrów:** Opcjonalne `GridSearchCV`.
* **Ewaluacja:** Zapis wyników do `lab2results.csv`.

### Dostępne komendy:
* `/classify dataset=<dataset_name> method=<model> gridsearch=<true/false> run=<n>`

---

## Etap 3: Laboratorium 3 - Głębokie Sieci Neuronowe i Analiza Sentymentu

Wprowadzenie asynchronicznych modeli Deep Learning z użyciem Keras/TensorFlow.

### Główne funkcjonalności:
* **Sieci Sekwencyjne:** Trening modeli `SimpleRNN`, `LSTM` oraz `GRU` na zbiorze IMDB.
* **MLOps:** Zapis wag (`.h5`) i tokenizerów (`.pkl`) z wykorzystaniem Lazy Loadingu dla oszczędności RAM.
* **Benchmarking:** Ewaluacja wariantów (Accuracy, F1, Macierze pomyłek) w konfrontacji z podejściem Transformerowym.

### Dostępne komendy:
* `/train <simplernn|lstm|gru> <dataset>` - Trening sieci.
* `/sentiment method=<metoda> text="<tekst>"` - Predykcja sentymentu.
* `/compare dataset=<baza> methods=<metoda1,metoda2>` - Ewaluacja porównawcza.

---

## Etap 4: Laboratorium 4 - Generatywne LLM, Translacja i NER/NEL

Integracja w pełni lokalnej, generatywnej sztucznej inteligencji.

### Główne funkcjonalności:
* **Streszczanie tekstów:** Silnik Ollama (Llama 3.2).
* **Translacja Offline:** Wykrywanie języka i tłumaczenie koder-dekoder (Helsinki-NLP).
* **Rozpoznawanie encji (NER):** Dwa silniki do wyboru - `spaCy` (polska fleksja) oraz neuronowa `Stanza`.
* **Linkowanie (NEL):** Automatyczna integracja z API Wikipedii w celu definiowania encji.

### Dostępne komendy:
* `/summarize type=<typ> length=<dlugosc> "tekst"`
* `/translate "tekst"`
* `/ner method=<spacy/stanza> "tekst"`

---

## Etap 5: Laboratorium 5 - Autonomiczny Agent AI i Multimodalność

Architektura agentowa z dostępem do zewnętrznych narzędzi (Function Calling).

### Główne funkcjonalności:
* **LLM Reasoning:** Agent (Llama 3.2) samodzielnie planuje, jakich narzędzi użyć do rozwiązania problemu.
* **Narzędzia (Tools):** Kalkulator, pogoda (OpenWeatherMap API), wyszukiwanie wiedzy (Wikipedia API - obejście Knowledge Cutoff).
* **Vision (LLaVA):** Model widzenia komputerowego - analiza przesyłanych na czacie obrazów.

### Dostępne komendy:
* `/ask [pytanie]` - Zadanie dla Agenta (np. pytania o pogodę, obliczenia).
* **[Wysłanie zdjęcia]** - Automatyczne parsowanie do modelu LLaVA.

---

## Etap 6: Laboratorium 6 - Content Moderation & Policy Enforcement (W trakcie realizacji)

System decyzyjny (Ensemble Model) chroniący przed spamem, danymi wrażliwymi (PII) i toksycznymi treściami.

### Główne funkcjonalności:
* **Privacy Filter Layer:** Implementacja modelu `openai/privacy-filter` (Hugging Face) do detekcji adresów, telefonów i danych kart kredytowych z wykorzystaniem progu ufności.
* **Pamięć trwała:** Baza oparta na plikach `.csv` (`moderation_log.csv`, `user_history.csv`) do logowania akcji moderacyjnych.
* **Optymalizacja zasobów:** Wdrożenie biblioteki `accelerate` wymuszającej `low_cpu_mem_usage` dla stabilnego ładowania dużych modeli.

---

## Wymagania Wstępne i Uruchomienie

1. **Klonowanie i wirtualne środowisko:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # lub venv\Scripts\activate w Windows
   pip install -r requirements.txt
