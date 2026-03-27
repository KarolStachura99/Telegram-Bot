import shlex
import os
import json
import nltk
import traceback
import numpy as np
from collections import Counter
from telegram import Update
from telegram.ext import ContextTypes
from nltk.util import ngrams
from sklearn.metrics import accuracy_score, f1_score

# --- IMPORTY Z LAB 1 ---
from bot.utils import save_sentence_to_json
from nlp.text_processing import (
    tokenize_text, remove_stopwords_from_tokens, 
    stem_tokens, lemmatize_tokens, get_bag_of_words, get_tfidf
)
from nlp.visualization import plot_histogram, plot_wordcloud, plot_bar_chart
from ml.classifier import train_and_predict

# --- IMPORTY Z LAB 2 ---
from ml.data_loader import load_dataset
from ml.vectorizers import vectorize_texts
from ml.models import train_model
from ml.visualization import (
    plot_wordclouds, plot_confusion_matrix, plot_embeddings, 
    save_feature_importance, save_results, save_similar_words_and_plot
)

# Upewniamy się, że NLTK umie dzielić tekst na zdania i tokenizować
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


# ==========================================
# KOMENDY Z LABORATORIUM 1
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa powitania."""
    await update.message.reply_text("Cześć! Jestem Botem NLP. Gotowy do zaawansowanych eksperymentów!")


async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa komendy /task do pojedynczych zadań NLP."""
    raw_text = update.message.text.replace("„", '"').replace("”", '"').replace("“", '"')
    try:
        args = shlex.split(raw_text)
    except ValueError:
        await update.message.reply_text("Błąd formatowania wiadomości. Pamiętaj o zamykaniu cudzysłowów!")
        return

    if len(args) < 4:
        await update.message.reply_text('Niepoprawne użycie! Poprawny format to:\n/task <nazwa_zadania> "tekst" "klasa"')
        return

    task_name = args[1]
    text_to_process = args[2]
    text_class = args[3]
    save_sentence_to_json(text_to_process, text_class)

    result_text = ""
    if task_name == "tokenize":
        tokens = tokenize_text(text_to_process)
        result_text = f"Wynik tokenizacji:\n{tokens}"
    elif task_name == "remove_stopwords":
        tokens = tokenize_text(text_to_process)
        cleaned_tokens = remove_stopwords_from_tokens(tokens)
        result_text = f"Oryginalne ({len(tokens)}): {tokens}\nBez stop words ({len(cleaned_tokens)}): {cleaned_tokens}"
    elif task_name == "stemming":
        tokens = tokenize_text(text_to_process)
        stemmed = stem_tokens(tokens)
        result_text = f"Wynik stemmingu:\n{stemmed}"
    elif task_name == "lemmatize":
        tokens = tokenize_text(text_to_process)
        lemmatized = lemmatize_tokens(tokens)
        result_text = f"Wynik lematyzacji:\n{lemmatized}"
    elif task_name == "bow":
        bow_result = get_bag_of_words(text_to_process)
        result_text = f"Reprezentacja Bag of Words:\n{bow_result}"
    elif task_name == "tfidf":
        tfidf_result = get_tfidf(text_to_process)
        result_text = f"Reprezentacja TF-IDF:\n{tfidf_result}"
    elif task_name == "plot_histogram":
        tokens = tokenize_text(text_to_process)
        filepath = plot_histogram(tokens)
        await update.message.reply_photo(photo=open(filepath, 'rb'), caption="✅ Zapisano do bazy.\nOto Twój histogram długości tokenów.")
        return
    elif task_name == "plot_wordcloud":
        tokens = tokenize_text(text_to_process)
        cleaned_tokens = remove_stopwords_from_tokens(tokens)
        filepath = plot_wordcloud(cleaned_tokens)
        await update.message.reply_photo(photo=open(filepath, 'rb'), caption="✅ Zapisano do bazy.\nOto Twoja chmura słów.")
        return
    else:
        result_text = f"Zadanie '{task_name}' nie jest zaimplementowane."

    response_message = f"✅ Zapisano do bazy jako '{text_class}'.\n\n🤖 Wykonano zadanie: {task_name}\n{result_text}"
    await update.message.reply_text(response_message)


async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa pełnego potoku przetwarzania tekstu (Lab 1)."""
    raw_text = update.message.text.replace("„", '"').replace("”", '"').replace("“", '"')
    try:
        args = shlex.split(raw_text)
    except ValueError:
        await update.message.reply_text("Błąd cudzysłowów!")
        return

    if len(args) < 3:
        await update.message.reply_text('Użycie: /full_pipeline "tekst" "klasa"')
        return

    text_to_process = args[1]
    text_class = args[2]

    tokens = tokenize_text(text_to_process)
    cleaned = remove_stopwords_from_tokens(tokens)
    lemmatized = lemmatize_tokens(cleaned)
    stemmed = stem_tokens(cleaned)
    bow = get_bag_of_words(text_to_process)
    tfidf = get_tfidf(text_to_process)

    hist_path = plot_histogram(tokens)
    wc_path = plot_wordcloud(cleaned)

    sentences = nltk.sent_tokenize(text_to_process)
    for sentence in sentences:
        save_sentence_to_json(sentence, text_class)

    raport = (
        f"🚀 **PEŁNY PIPELINE ZAKOŃCZONY**\n\n"
        f"Liczba wydzielonych zdań zapisanych do bazy: {len(sentences)}\n\n"
        f"🔹 **Tokeny:** {tokens[:5]}...\n"
        f"🔹 **Bez stopwords:** {cleaned[:5]}...\n"
        f"🔹 **Lematyzacja:** {lemmatized[:5]}...\n"
        f"🔹 **Stemming:** {stemmed[:5]}...\n"
        f"🔹 **BoW:** {bow[:100]}...\n"
        f"🔹 **TF-IDF:** {tfidf[:100]}..."
    )
    
    await update.message.reply_text(raport, parse_mode='Markdown')
    await update.message.reply_photo(photo=open(hist_path, 'rb'), caption="Histogram długości tokenów")
    await update.message.reply_photo(photo=open(wc_path, 'rb'), caption="Word Cloud")


async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa klasyfikatora pojedynczych wiadomości (Lab 1)."""
    raw_text = update.message.text.replace("„", '"').replace("”", '"').replace("“", '"')
    try:
        args = shlex.split(raw_text)
    except ValueError:
        await update.message.reply_text("Błąd cudzysłowów!")
        return

    if len(args) < 2:
        await update.message.reply_text('Użycie: /classifier "tekst do sprawdzenia"')
        return

    text_to_test = args[1]
    predicted_class = train_and_predict(text_to_test)
    
    await update.message.reply_text(
        f"🤖 **Wynik klasyfikacji:**\n\nTekst: '{text_to_test}'\nPrzewidziana klasa: **{predicted_class.upper()}**",
        parse_mode='Markdown'
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa statystyk dla pojedynczych zdań z json (Lab 1)."""
    if not os.path.exists("data/sentences.json"):
        await update.message.reply_text("Baza danych jest pusta. Dodaj teksty przez /task.")
        return
    with open("data/sentences.json", "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            await update.message.reply_text("Błąd pliku bazy danych.")
            return
    if not data:
        await update.message.reply_text("Baza jest pusta.")
        return
        
    all_text = " ".join([item["text"] for item in data])
    classes = [item["class"] for item in data]
    class_counts = dict(Counter(classes))
    tokens = tokenize_text(all_text)
    cleaned = remove_stopwords_from_tokens(tokens)
    
    bigrams = list(ngrams(cleaned, 2))
    trigrams = list(ngrams(cleaned, 3))
    unique_tokens = len(set(cleaned))
    
    bar_path = plot_bar_chart(cleaned)
    hist_path = plot_histogram(cleaned)
    wc_path = plot_wordcloud(cleaned)
    
    raport = (
        f"📊 **STATYSTYKI ZBIORU DANYCH**\n\n"
        f"Liczba wszystkich tekstów: {len(data)}\n"
        f"Liczność klas: {class_counts}\n"
        f"Unikalne tokeny (bez stopwords): {unique_tokens}\n"
        f"Przykładowe unikalne 2-gramy: {list(set(bigrams))[:3]}\n"
        f"Przykładowe unikalne 3-gramy: {list(set(trigrams))[:3]}"
    )
    
    await update.message.reply_text(raport, parse_mode='Markdown')
    await update.message.reply_photo(photo=open(bar_path, 'rb'), caption="Wykres najczęstszych słów")
    await update.message.reply_photo(photo=open(hist_path, 'rb'), caption="Histogram długości")
    await update.message.reply_photo(photo=open(wc_path, 'rb'), caption="Word Cloud")


# ==========================================
# KOMENDA Z LABORATORIUM 2
# ==========================================

async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa zaawansowanego eksperymentu klasyfikacji zbiorów danych (Lab 2)."""
    await update.message.reply_text("⏳ Rozpoczynam parsowanie parametrów eksperymentu...")
    
    # Odczytywanie parametrów z wiadomości
    args = context.args
    params = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            params[key.lower()] = value.lower()

    # Weryfikacja parametrów (ustawianie domyślnych, jeśli brakuje)
    dataset_name = params.get("dataset", "20news_group")
    method_param = params.get("method", "nb")
    gridsearch_str = params.get("gridsearch", "false")
    gridsearch = True if gridsearch_str == "true" else False
    try:
        runs = int(params.get("run", "1"))
    except ValueError:
        runs = 1

    # Wysłanie powiadomienia do użytkownika (Używamy HTML zamiast Markdown dla bezpieczeństwa znaków!)
    await update.message.reply_text(
        f"⚙️ <b>Konfiguracja eksperymentu:</b>\n"
        f"Dataset: {dataset_name}\n"
        f"Modele: {method_param}\n"
        f"GridSearch: {gridsearch}\n"
        f"Liczba uruchomień: {runs}",
        parse_mode="HTML"
    )

    try:
        # 1. Wczytanie danych
        await update.message.reply_text(f"📥 Pobieranie zbioru danych: {dataset_name}...")
        texts, labels, class_names = load_dataset(dataset_name)
        
        # Generowanie chmur słów (tylko raz dla całego korpusu i klas)
        await update.message.reply_text("☁️ Generowanie chmur słów...")
        plot_wordclouds(texts, labels, class_names)

        # Predefiniowane seedy z instrukcji
        seeds = [42, 1337, 2024]
        
        # Embeddingi i modele do przetestowania
        embeddings_to_test = ['bow', 'tfidf', 'word2vec', 'glove']
        models_to_test = ['nb', 'rf', 'mlp', 'logreg'] if method_param == "all" else [method_param]

        # 2. Główna pętla eksperymentu
        for run_idx in range(min(runs, len(seeds))):
            current_seed = seeds[run_idx]
            await update.message.reply_text(f"🚀 <b>Rozpoczynam Run {run_idx+1}/{runs}</b> (Seed: {current_seed})", parse_mode="HTML")
            
            for emb in embeddings_to_test:
                await update.message.reply_text(f"🧮 Wektoryzacja metodą: {emb.upper()}...")
                X = vectorize_texts(emb, texts)
                
                # Dodatkowe zadanie: podobne słowa dla modeli gęstych
                if emb in ['word2vec', 'glove'] and run_idx == 0:
                    try:
                        import gensim.downloader as api
                        from gensim.models import Word2Vec
                        import nltk
                        model_for_words = api.load("glove-wiki-gigaword-50") if emb == 'glove' else Word2Vec([nltk.word_tokenize(t.lower()) for t in texts], vector_size=100, min_count=1).wv
                        save_similar_words_and_plot(model_for_words, ['space', 'computer', 'science', 'music', 'car'])
                    except Exception as e:
                        print(f"Błąd przy podobnych słowach: {e}")

                for mod in models_to_test:
                    await update.message.reply_text(f"🧠 Trenowanie modelu: {mod.upper()} na danych {emb.upper()}...")
                    
                    trained_model = train_model(mod, X, labels, gridsearch, current_seed)
                    y_pred = trained_model.predict(X)
                    
                    acc = accuracy_score(labels, y_pred)
                    mac_f1 = f1_score(labels, y_pred, average='macro')
                    
                    # Zapisywanie do pliku .csv
                    save_results(emb, mod, acc, mac_f1, current_seed)
                    
                    # Wizualizacje macierzy i embeddingów
                    plot_confusion_matrix(labels, y_pred, class_names, emb, mod)
                    plot_embeddings(X, labels, dataset_name, mod, emb)
                    
                    # Analiza ważności cech (tylko dla BoW/TFIDF i wybranych modeli)
                    if emb in ['bow', 'tfidf'] and mod in ['nb', 'rf']:
                        from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
                        vec_obj = CountVectorizer(max_features=5000) if emb == 'bow' else TfidfVectorizer(max_features=5000)
                        vec_obj.fit(texts)
                        save_feature_importance(trained_model, vec_obj, class_names, dataset_name)

        await update.message.reply_text(
            "✅ <b>Eksperyment zakończony sukcesem!</b>\n"
            "Wszystkie wyniki zostały zapisane w lab2results.csv.\n"
            "Wykresy znajdziesz w folderze lab2plots/.",
            parse_mode="HTML"
        )
        
    except Exception as e:
        error_msg = f"❌ Wystąpił błąd podczas eksperymentu:\n{str(e)}\n\n{traceback.format_exc()}"
        # Telegram blokuje wiadomości powyżej ok. 4000 znaków, ucinamy błąd w razie czego
        await update.message.reply_text(error_msg[:4000])
        print(error_msg)