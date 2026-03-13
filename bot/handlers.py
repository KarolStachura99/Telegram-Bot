import shlex
import os
import json
import nltk
from collections import Counter
from telegram import Update
from telegram.ext import ContextTypes
from nltk.util import ngrams
from nlp.visualization import plot_bar_chart

# Nasze własne moduły
from bot.utils import save_sentence_to_json
from nlp.text_processing import (
    tokenize_text, remove_stopwords_from_tokens, 
    stem_tokens, lemmatize_tokens, get_bag_of_words, get_tfidf
)
from nlp.visualization import plot_histogram, plot_wordcloud, plot_bar_chart
from ml.classifier import train_and_predict

# Upewniamy się, że NLTK umie dzielić tekst na zdania (potrzebne do full_pipeline)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa powitania."""
    await update.message.reply_text("Cześć! Jestem Botem NLP z laboratorium. Gotowy do pracy!")


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

    # Zapis do JSON (wymóg z laboratorium)
    save_sentence_to_json(text_to_process, text_class)

    result_text = ""

    # Logika wyboru zadania NLP
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
        await update.message.reply_photo(
            photo=open(filepath, 'rb'), 
            caption=f"✅ Zapisano do bazy.\nOto Twój histogram długości tokenów."
        )
        return
        
    elif task_name == "plot_wordcloud":
        tokens = tokenize_text(text_to_process)
        cleaned_tokens = remove_stopwords_from_tokens(tokens)
        filepath = plot_wordcloud(cleaned_tokens)
        await update.message.reply_photo(
            photo=open(filepath, 'rb'), 
            caption=f"✅ Zapisano do bazy.\nOto Twoja chmura słów."
        )
        return
        
    else:
        result_text = f"Zadanie '{task_name}' nie jest jeszcze zaimplementowane."

    response_message = (
        f"✅ Zapisano do bazy jako '{text_class}'.\n\n"
        f"🤖 Wykonano zadanie: {task_name}\n"
        f"{result_text}"
    )
    
    await update.message.reply_text(response_message)


async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa pełnego potoku przetwarzania tekstu."""
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

    # 1. Wykonanie operacji NLP
    tokens = tokenize_text(text_to_process)
    cleaned = remove_stopwords_from_tokens(tokens)
    lemmatized = lemmatize_tokens(cleaned)
    stemmed = stem_tokens(cleaned)
    bow = get_bag_of_words(text_to_process)
    tfidf = get_tfidf(text_to_process)

    # 2. Generowanie wykresów
    hist_path = plot_histogram(tokens)
    wc_path = plot_wordcloud(cleaned)

    # 3. Podział na zdania i naiwne przypisanie klas (wymóg laboratorium)
    sentences = nltk.sent_tokenize(text_to_process)
    for sentence in sentences:
        save_sentence_to_json(sentence, text_class)

    # 4. Wysyłanie wyników do użytkownika
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
    """Obsługa klasyfikatora (uczenie maszynowe)."""
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
    
    # Uruchomienie modelu na podstawie danych z JSON
    predicted_class = train_and_predict(text_to_test)
    
    await update.message.reply_text(
        f"🤖 **Wynik klasyfikacji:**\n\n"
        f"Tekst: '{text_to_test}'\n"
        f"Przewidziana klasa: **{predicted_class.upper()}**",
        parse_mode='Markdown'
    )


async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa globalnych statystyk całego zbioru."""
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
        
    # Zbieramy wszystkie teksty i klasy w jedną całość
    all_text = " ".join([item["text"] for item in data])
    classes = [item["class"] for item in data]
    
    # 1. Liczność klas
    class_counts = dict(Counter(classes))
    
    # 2. Czyszczenie do statystyk (żeby wykresy miały sens)
    tokens = tokenize_text(all_text)
    cleaned = remove_stopwords_from_tokens(tokens)
    
    # 3. N-gramy i tokeny
    bigrams = list(ngrams(cleaned, 2))
    trigrams = list(ngrams(cleaned, 3))
    unique_tokens = len(set(cleaned))
    
    # 4. Wykresy
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
    await update.message.reply_photo(photo=open(wc_path, 'rb'), caption="Word Cloud dla całego zbioru")