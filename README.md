# Telegram Bot - NLP & Text Classification (Laboratorium 1)

Moduł realizowany w ramach Etapu 1 (Laboratorium 1). W tym kroku zaimplementowano podstawowy rurociąg (pipeline) NLP, pozwalający na analizę i klasyfikację pojedynczych wiadomości tekstowych przesyłanych do bota za pomocą komunikatora Telegram.

## Główne funkcjonalności (Lab 1):
* **Przetwarzanie tekstu (Preprocessing):** Tokenizacja, usuwanie stop words, stemming oraz lematyzacja.
* **Wektoryzacja:** Zamiana tekstu na reprezentacje liczbowe za pomocą Bag of Words (BoW) oraz TF-IDF.
* **Wizualizacja:** Generowanie chmur słów (Word Cloud) oraz histogramów długości tokenów dla podanych zdań (zapisywane w katalogu `plots/`).
* **Baza danych:** Zapisywanie przetworzonych zdań i ich klas do pliku `data/sentences.json`.

## Dostępne komendy (Lab 1):
* `/start` - Powitanie i inicjalizacja bota.
* `/task <zadanie> "tekst" "klasa"` - Wykonanie pojedynczego zadania NLP (np. `tokenize`, `stemming`, `bow`, `plot_wordcloud`).
* `/full_pipeline "tekst" "klasa"` - Przeprowadzenie pełnego procesu przetwarzania tekstu i wygenerowanie raportu.
* `/classifier "tekst"` - Szybka klasyfikacja podanej wiadomości.
* `/stats` - Generowanie statystyk i wykresów dla całego zgromadzonego zbioru danych.

## Uruchomienie
Środowisko należy wygenerować lokalnie.

1. Sklonuj repozytorium lub rozpakuj archiwum `.zip`.
2. Utwórz wirtualne środowisko w głównym folderze projektu: 
   `python -m venv venv`
3. Aktywuj środowisko:
   * Windows: `venv\Scripts\activate`
   * Mac/Linux: `source venv/bin/activate`
4. Zainstaluj wymagane biblioteki: 
   `pip install -r requirements.txt`
5. Skonfiguruj klucz API Telegrama w pliku `.env` lub `config.py`.
6. Uruchom aplikację: 
   `python main.py`