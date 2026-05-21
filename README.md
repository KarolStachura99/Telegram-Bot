# Telegram Bot - NLP & Text Classification (Laboratorium 1 & 2)

Projekt realizowany w ramach Etapu 2. Zaawansowany bot na platformę Telegram służący do przetwarzania języka naturalnego (NLP) oraz przeprowadzania zautomatyzowanych eksperymentów z zakresu uczenia maszynowego na dużych zbiorach danych tekstowych.

## Etap 1: Przetwarzanie tekstu (Lab 1)
* **Pre-processing:** Tokenizacja, usuwanie stop words, stemming, lematyzacja.
* **Komendy:** `/task`, `/full_pipeline`, `/classifier`, `/stats`.

## Etap 2: Eksperymenty Klasyfikacji (Lab 2)
W tym module rozbudowano architekturę o zautomatyzowane eksperymenty badawcze:
* **Wektoryzacja i Embeddingi:** Implementacja BoW, TF-IDF oraz gęstych reprezentacji (Word2Vec, GloVe).
* **Modele ML (scikit-learn):** Naive Bayes, Random Forest, MLPClassifier, Logistic Regression.
* **Optymalizacja:** Wbudowane strojenie hiperparametrów za pomocą `GridSearchCV`.
* **Ewaluacja:** Zapis logów do `lab2results.csv` oraz generowanie macierzy pomyłek, rzutów wektorowych (PCA/t-SNE) i analizy cech (Top 10 Feature Importance) w folderze `lab2plots/`.

## Dostępne komendy badawcze (Lab 2):
Eksperymenty uruchamiane są za pomocą parametryzowanej komendy:
`/classify dataset=<dataset_name> method=<model> gridsearch=<true/false> run=<n>`
*Przykład:* `/classify dataset=20news_group method=rf gridsearch=false run=1`

## Uuchomienie
Środowisko należy wygenerować lokalnie na podstawie pliku `requirements.txt`.

1. Rozpakuj archiwum `.zip` lub sklonuj repozytorium.
2. Utwórz wirtualne środowisko w folderze projektu: `python -m venv venv`
3. Aktywuj środowisko (`venv\Scripts\activate` lub `source venv/bin/activate`).
4. Zainstaluj wymagane biblioteki: `pip install -r requirements.txt`
5. Utwórz plik `.env` i dodaj swój token: `TELEGRAM_TOKEN=twój_token`
6. Uruchom aplikację: `python main.py`