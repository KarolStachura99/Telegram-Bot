# Telegram Bot - NLP & Text Classification

Zaawansowany bot na platformę Telegram, zbudowany w języku Python, służący do przetwarzania języka naturalnego (NLP) oraz przeprowadzania eksperymentów z zakresu uczenia maszynowego (Machine Learning), głębokich sieci neuronowych (Deep Learning) i generatywnej sztucznej inteligencji. Projekt został zrealizowany w ramach czterech etapów laboratoryjnych.

---

## Etap 1: Laboratorium 1 - Przetwarzanie tekstu i podstawowa klasyfikacja

W pierwszym etapie zaimplementowano podstawowy rurociąg (pipeline) NLP, pozwalający na analizę i klasyfikację pojedynczych wiadomości tekstowych przesyłanych do bota.

### Główne funkcjonalności:
* **Przetwarzanie tekstu (Preprocessing):** Tokenizacja, usuwanie stop words, stemming oraz lematyzacja.
* **Wektoryzacja:** Zamiana tekstu na reprezentacje liczbowe za pomocą Bag of Words (BoW) oraz TF-IDF.
* **Wizualizacja:** Generowanie chmur słów (Word Cloud) oraz histogramów długości tokenów dla podanych zdań (zapisywane w katalogu `plots/`).
* **Baza danych:** Zapisywanie przetworzonych zdań i ich klas do pliku `data/sentences.json`.

### Dostępne komendy (Lab 1):
* `/start` - Powitanie i inicjalizacja bota.
* `/task <zadanie> "tekst" "klasa"` - Wykonanie pojedynczego zadania NLP (np. `tokenize`, `stemming`, `bow`, `plot_wordcloud`).
* `/full_pipeline "tekst" "klasa"` - Przeprowadzenie pełnego procesu przetwarzania tekstu i wygenerowanie raportu.
* `/classifier "tekst"` - Szybka klasyfikacja podanej wiadomości.
* `/stats` - Generowanie statystyk i wykresów dla całego zgromadzonego zbioru danych.

---

## Etap 2: Laboratorium 2 - Eksperymenty klasyfikacji dla całych zbiorów danych

Drugi etap rozszerza architekturę o moduł badawczy do przeprowadzania zautomatyzowanych eksperymentów na dużych zbiorach danych tekstowych. 

### Główne funkcjonalności:
* **Obsługa datasetów:** Zautomatyzowane pobieranie i ładowanie korpusów tekstowych (m.in. `20news_group`).
* **Zaawansowane embeddingi:** Implementacja wektoryzacji za pomocą TF-IDF, BoW, a także gęstych osadzeń: modelu Word2Vec oraz pretrenowanego modelu GloVe.
* **Modele klasyfikacji:** Możliwość treningu modeli: Multinomial Naive Bayes (`nb`), Random Forest (`rf`), MLPClassifier (`mlp`) oraz Logistic Regression (`logreg`).
* **Strojenie hiperparametrów:** Opcjonalne uruchamianie `GridSearchCV`.
* **Ewaluacja i raportowanie:** Zapis wyników do głównego pliku `lab2results.csv`.

### Dostępne komendy (Lab 2):
Eksperymenty uruchamiane są za pomocą jednej, parametryzowanej komendy:
`/classify dataset=<dataset_name> method=<model> gridsearch=<true/false> run=<n>`

---

## Etap 3: Laboratorium 3 - Głębokie Sieci Neuronowe i Transformery (Sentyment)

Trzeci etap wprowadza asynchroniczne modele głębokiego uczenia (Deep Learning) z wykorzystaniem biblioteki Keras/TensorFlow oraz najnowocześniejsze modele językowe ze środowiska Hugging Face.

### Główne funkcjonalności:
* **Trening Sieci Sekwencyjnych:** Możliwość trenowania od zera modeli `SimpleRNN`, `LSTM` oraz `GRU` na zewnętrznych zbiorach danych (IMDB).
* **Zarządzanie modelami (MLOps):** Zapis wytrenowanych wag do plików `.h5` oraz tokenizerów do `.pkl`, wraz ze zjawiskiem Lazy Loadingu zapobiegającym wyciekom pamięci RAM.
* **Optymalizacja sieci:** Implementacja warstwy `Dropout` (0.5) w celu eliminacji zjawiska przeuczenia (overfittingu).
* **Zaawansowana ewaluacja (Benchmarking):** Porównywanie sieci sekwencyjnych z podejściami klasycznymi (TextBlob, Stanza, Rule-based) oraz potężnymi Transformerami (pipeline `sentiment-analysis`).
* **Generowanie artefaktów:** Rysowanie macierzy pomyłek (Confusion Matrix) biblioteką `seaborn` oraz zrzut logów skuteczności do `lab3results.csv`.

### Dostępne komendy (Lab 3):
* `/train <simplernn|lstm|gru> <dataset>` - Trenuje wybraną sieć na wskazanym zbiorze i rysuje wykres skuteczności.
* `/sentiment method=<metoda> text="<tekst>" [dataset=<baza>]` - Predykcja sentymentu na żywo za pomocą wczytanego z dysku modelu.
* `/compare dataset=<baza> methods=<metoda1,metoda2>` - Przeprowadza pełną ewaluację (Accuracy, Precision, Recall, F1) na zbiorze i rysuje Macierze Pomyłek.
* `/add_sentiment "<tekst>" "<etykieta>"` - Zapisuje nowe zdanie do własnej bazy danych.
* `/models` - Wyświetla listę wgranych i gotowych do użycia modeli `.h5`.
* `/help` - Wyświetla pełną listę komend z poziomu komunikatora.

---

## Etap 4: Laboratorium 4 - Generatywne LLM, Translacja i Ekstrakcja Bytów (NER/NEL)

Czwarty etap poszerza bota o możliwości w pełni lokalnej, generatywnej sztucznej inteligencji, głębokie modele koder-dekoder oraz łączenie tekstu z zewnętrznymi bazami wiedzy.

### Główne funkcjonalności:
* **Streszczanie tekstu (Lokalne LLM):** Asynchroniczna integracja z silnikiem Ollama i modelem Llama 3.2. Wdrożenie System Promptów i reguł Prompt Engineeringu zapobiegających halucynacjom modelu.
* **Tłumacz Offline (Transformers):** Tłumaczenie języków wspierane biblioteką `langdetect` z użyciem modeli koder-dekoder (Helsinki-NLP / Hugging Face). Implementacja bezpiecznego, ręcznego potoku wywołującego `AutoTokenizer` oraz `AutoModelForSeq2SeqLM`.
* **Rozpoznawanie Bytów (NER):** Implementacja dwóch niezależnych i konkurencyjnych podejść analitycznych:
  * Lekkiego potoku statystycznego z użyciem **spaCy** (model `pl_core_news_lg` zoptymalizowany pod polską fleksję).
  * Złożonych modeli neuronowych z Uniwersytetu Stanforda z użyciem **Stanza** (mapowanie standardów NKJP, precyzyjna tokenizacja MWT).
* **Linkowanie (NEL) i Bazy Wiedzy:** Algorytm automatycznej lematyzacji nazw własnych, nawiązujący połączenie z API Wikipedii w celu pobierania i rozwiązywania konfliktów (NED) kontekstowych dla wyodrębnionych encji.

### Dostępne komendy (Lab 4):
* `/summarize type=<bullets/abstractive> length=<short/medium/long> "tekst"` - Streszczenie wskazanego tekstu za pomocą modelu Llama 3.2.
* `/translate "tekst"` - Wykrycie języka i tłumaczenie offline (PL/EN).
* `/ner method=<spacy/stanza> "tekst"` - Analiza obiektów nazwanych i generowanie linków do Wikipedii przy użyciu wybranego silnika.

---

## Uruchomienie

**Środowisko należy wygenerować lokalnie na podstawie pliku `requirements.txt`. Upewnij się, że spełnione są wszystkie wymagania systemowe dla modeli offline.**

### Krok 1: Podstawowa instalacja
1. Sklonuj repozytorium na swój dysk lokalny lub rozpakuj archiwum `.zip`.
2. Utwórz wirtualne środowisko w głównym folderze projektu: 
   `python -m venv venv`
3. Aktywuj środowisko:
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. Zainstaluj wszystkie wymagane biblioteki: 
   `pip install -r requirements.txt`

### Krok 2: Instalacja zasobów dla Lab 4 (Zewnętrzne modele)
1. **Model spaCy dla języka polskiego:** W aktywowanym środowisku wirtualnym pobierz duży model analityczny:  
   `python -m spacy download pl_core_news_lg`
2. **Model LLM (Ollama):** Zainstaluj oprogramowanie [Ollama](https://ollama.com/), a następnie w standardowym terminalu pobierz model używany do streszczeń (ok. 2 GB):  
   `ollama pull llama3.2`  
   *(Stanza oraz modele Helsinki-NLP pobierają swoje wagi automatycznie przy pierwszym wywołaniu odpowiedniej komendy przez bota).*

### Krok 3: Konfiguracja i uruchomienie
1. Skonfiguruj klucz API Telegrama (w zależności od implementacji w kodzie, zaktualizuj plik `.env` lub `config.py`):
   `TELEGRAM_TOKEN=twój_token_od_botfather`
2. Uruchom aplikację: 
   `python main.py`