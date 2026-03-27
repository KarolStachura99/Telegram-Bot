# Telegram Bot - NLP & Text Classification

Zaawansowany bot na platformę Telegram, zbudowany w języku Python, służący do przetwarzania języka naturalnego (NLP) oraz przeprowadzania eksperymentów z zakresu uczenia maszynowego (Machine Learning). Projekt został zrealizowany w ramach dwóch etapów laboratoryjnych.

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
* **Zaawansowane embeddingi:** Implementacja wektoryzacji za pomocą TF-IDF, BoW, a także gęstych osadzeń: modelu Word2Vec (trenowanego na korpusie) oraz pretrenowanego modelu GloVe.
* **Modele klasyfikacji:** Możliwość treningu modeli: Multinomial Naive Bayes (`nb`), Random Forest (`rf`), MLPClassifier (`mlp`) oraz Logistic Regression (`logreg`).
* **Strojenie hiperparametrów:** Opcjonalne uruchamianie `GridSearchCV` w celu znalezienia optymalnych parametrów dla wybranych klasyfikatorów.
* **Ewaluacja i raportowanie:** Zapis wyników eksperymentów (Accuracy, Macro F1, użyty Seed) do głównego pliku `lab2results.csv`.

### Generowane artefakty badawcze (katalog `lab2plots/`):
* **Wizualizacja przestrzeni wektorowej:** Rzuty embeddingów wykonane algorytmami redukcji wymiarowości: PCA, t-SNE oraz TruncatedSVD.
* **Analiza modeli:** Macierze pomyłek (Confusion Matrix) dla każdego testowanego wariantu.
* **Analiza cech:** Zapis najważniejszych cech (Top 10 Feature Importance) dla algorytmów opartych na BoW/TF-IDF.
* **Podobne słowa:** Plik `lab2_similar_words.txt` oraz wykresy obrazujące relacje semantyczne wyciągnięte z modeli Word2Vec/GloVe.
* **Chmury słów:** Generowane dla całego korpusu oraz niezależnie dla każdej klasy decyzyjnej.

### Dostępne komendy (Lab 2):
Eksperymenty uruchamiane są za pomocą jednej, parametryzowanej komendy:
`/classify dataset=<dataset_name> method=<model> gridsearch=<true/false> run=<n>`

**Przykłady użycia:**
* `/classify dataset=20news_group method=all gridsearch=false run=3`
* `/classify dataset=20news_group method=nb gridsearch=true run=1`

---

## Wymagania i uruchomienie

1. Sklonuj repozytorium na swój dysk lokalny.
2. Utwórz wirtualne środowisko: `python -m venv venv`
3. Aktywuj środowisko i zainstaluj zależności: `pip install -r requirements.txt`
4. Utwórz plik `.env` w głównym katalogu projektu i dodaj w nim swój token bota:
   `TELEGRAM_TOKEN=twoj_token_tutaj`
5. Uruchom bota: `python main.py`
