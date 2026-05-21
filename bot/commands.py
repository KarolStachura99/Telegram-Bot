import asyncio
import os
import shlex
import matplotlib.pyplot as plt
from telegram import Update
from telegram.ext import ContextTypes
import config

# ==========================================
# LABORATORIUM 1: Przetwarzanie i Statystyki
# ==========================================

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "<b>Witaj w zaawansowanym systemie NLP!</b>\n\n"
        "Projekt obejmuje moduły z Lab 1, 2 i 3.\n"
        "Użyj /task, aby przetworzyć tekst, lub /compare, aby sprawdzić AI.",
        parse_mode="HTML"
    )

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from nlp.text_processing import tokenize_text, remove_stopwords_from_tokens, lemmatize_tokens
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("❌ Podaj tekst po komendzie.")
        return

    tokens = tokenize_text(text)
    no_stop = remove_stopwords_from_tokens(tokens)
    lemmas = lemmatize_tokens(no_stop)
    
    response = (
        f"<b>Przetworzony tekst:</b>\n"
        f"Tokeny: {tokens}\n"
        f"Po usunięciu stop words: {no_stop}\n"
        f"Lematy: {lemmas}"
    )
    await update.message.reply_text(response, parse_mode="HTML")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from nlp.text_processing import tokenize_text
    from nlp.visualization import plot_histogram, plot_wordcloud
    
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("❌ Podaj tekst do analizy statystycznej.")
        return

    tokens = tokenize_text(text)
    hist_path = plot_histogram(tokens)
    cloud_path = plot_wordcloud(tokens)

    with open(hist_path, 'rb') as h, open(cloud_path, 'rb') as c:
        await update.message.reply_photo(h, caption=" Histogram długości słów")
        await update.message.reply_photo(c, caption=" Chmura słów")

# ==========================================
# LABORATORIUM 2: Klasyfikacja i Modele ML
# ==========================================

async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from ml.classifier import train_and_predict
    text = " ".join(context.args)
    if not text:
        await update.message.reply_text("❌ Podaj tekst do klasyfikacji.")
        return

    result = await asyncio.to_thread(train_and_predict, text)
    await update.message.reply_text(f" Klasyfikacja modelu ML: <b>{result}</b>", parse_mode="HTML")

async def train_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from ml.training import train_sequential_model
    if len(context.args) < 2:
        await update.message.reply_text("❌ Format: /train [model_type] [dataset_name]")
        return

    model_type, dataset = context.args[0], context.args[1]
    msg = await update.message.reply_text(f" Trenowanie modelu {model_type} na zbiorze {dataset}...")
    
    try:
        path, plot, acc = await asyncio.to_thread(train_sequential_model, model_type, dataset)
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg.message_id)
        with open(plot, 'rb') as p:
            await update.message.reply_photo(p, caption=f"✅ Trening zakończony!\nAccuracy: {acc:.2f}\nModel: {path}")
    except Exception as e:
        await update.message.reply_text(f"❌ Błąd treningu: {str(e)}")

# ==========================================
# LABORATORIUM 3: Sentyment i Transformery
# ==========================================

async def add_sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from ml.datasets import add_to_custom_dataset
    raw_text = " ".join(context.args)
    try:
        args = shlex.split(raw_text)
        if len(args) != 2:
            await update.message.reply_text("❌ Format: /add_sentiment \"tekst\" \"etykieta\"")
            return
        add_to_custom_dataset(args[0], args[1])
        await update.message.reply_text(f"✅ Dodano do bazy: <b>{args[0]}</b>", parse_mode="HTML")
    except Exception as e:
        await update.message.reply_text(f"❌ Błąd: {str(e)}")

async def compare_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from ml.datasets import load_dataset
    from ml.evaluation import evaluate_and_plot_matrix, save_results_to_csv
    from ml.model_loader import predict_sequential
    from ml.sentiment_methods import analyze_sentiment
    import asyncio

    raw_args = " ".join(context.args)

    # 1. Parsowanie parametrów
    if "dataset=" not in raw_args or "methods=" not in raw_args:
        await update.message.reply_text(" Użyj formatu: /compare dataset=<nazwa> methods=<metoda1,metoda2>")
        return

    try:
        dataset_name = raw_args.split("dataset=")[1].split(" ")[0].strip().lower()
        methods_str = raw_args.split("methods=")[1].split(" ")[0].strip().lower()
        methods = methods_str.split(",")

        msg = await update.message.reply_text(f" Rozpoczynam potężną ewaluację na zbiorze {dataset_name} dla metod: {', '.join(methods)}.\nTo może potrwać...")

        # 2. Pobranie danych do testów
        texts, labels, _ = await asyncio.to_thread(load_dataset, dataset_name)
        
        # BEZPIECZNIK WYDAJNOŚCI: Skracamy zbiór do 30 elementów, żeby nie spalić procesora
        if len(texts) > 30:
            texts = texts[:30]
            labels = labels[:30]

        results_list = []
        photos_to_send = []

        # 3. Główna pętla sprawdzająca modele
        for method in methods:
            method = method.strip()
            y_true = []
            y_pred = []
            
            # Przetwarzanie każdego tekstu ze zbioru
            for i in range(len(texts)):
                text = texts[i]
                true_label = labels[i].lower() # np. "pozytywny"
                
                try:
                    # Rozgałęzienie: modele Keras vs klasyczne NLP
                    if method in ["simplernn", "lstm", "gru"]:
                        pred_label, _ = await asyncio.to_thread(predict_sequential, method, text, dataset_name)
                    else:
                        pred_label, _ = await asyncio.to_thread(analyze_sentiment, method, text)
                        
                    y_true.append(true_label)
                    y_pred.append(pred_label)
                except Exception as eval_err:
                    print(f"Błąd predykcji dla {method}: {eval_err}")
                    continue
            
            # 4. Ewaluacja wyników i rysowanie macierzy
            if len(y_true) > 0:
                eval_data = await asyncio.to_thread(evaluate_and_plot_matrix, y_true, y_pred, method, dataset_name)
                results_list.append(eval_data)
                photos_to_send.append(eval_data["Matrix_Path"])

        # 5. Zapis wyników do pliku CSV
        if results_list:
            await asyncio.to_thread(save_results_to_csv, results_list)

        # 6. Formatowanie i wysyłka wiadomości końcowej
        summary = f"**Ewaluacja zakończona!**\nZbiór: {dataset_name} (próbka {len(texts)} el.)\nWyniki zapisano w `lab3results.csv`\n\n"
        for res in results_list:
            summary += f" **{res['Model']}** | Acc: {res['Accuracy']:.2f} | F1: {res['F1-Score']:.2f}\n"

        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg.message_id)
        await update.message.reply_text(summary, parse_mode="Markdown")

        # 7. Wysyłanie wygenerowanych macierzy pomyłek
        for photo_path in photos_to_send:
            with open(photo_path, 'rb') as photo:
                await update.message.reply_photo(photo=photo)

    except Exception as e:
        await update.message.reply_text(f"❌ Wystąpił błąd ewaluacji: {str(e)}")

# Aliasy dla kompatybilności z main.py
async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await task_command(update, context)

async def classify_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await classifier_command(update, context)

async def sentiment_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from ml.model_loader import predict_sequential
    import asyncio
    
    raw_args = " ".join(context.args)
    
    if "method=" not in raw_args or "text=" not in raw_args:
        await update.message.reply_text("❌ Zły format! Użyj: /sentiment method=<metoda> text=\"Twój tekst\" [dataset=<opcjonalnie>]")
        return
        
    try:
        # Dynamiczne parsowanie argumentów
        method = raw_args.split("method=")[1].split(" ")[0].strip().lower()
        
        # Pobieranie datasetu, domyślnie custom
        dataset = "custom"
        if "dataset=" in raw_args:
            dataset = raw_args.split("dataset=")[1].strip().lower()
            
        # Wyciąganie tekstu (bezpieczne oddzielenie od parametru dataset)
        text_part = raw_args.split("text=")[1]
        if "dataset=" in text_part:
            text_part = text_part.split("dataset=")[0]
            
        text = text_part.strip().strip('"').strip("'")
        
        msg = await update.message.reply_text(f"🔍 Analiza metodą {method.upper()} na zbiorze {dataset}...")
        
        # Delegacja do osobnego wątku
        label, confidence = await asyncio.to_thread(predict_sequential, method, text, dataset)
        
        response = (
            f"<b>Model:</b> {method.upper()}\n"
            f"<b>Dataset:</b> {dataset}\n"
            f"<b>Predykcja:</b> {label}\n"
            f"<b>Pewność:</b> {confidence:.2f}"
        )
        
        await context.bot.delete_message(chat_id=update.effective_chat.id, message_id=msg.message_id)
        await update.message.reply_text(response, parse_mode="HTML")
        
    except FileNotFoundError as e:
        await update.message.reply_text(f"❌ {str(e)}")
    except Exception as e:
        await update.message.reply_text(f"❌ Wystąpił błąd podczas analizy: {str(e)}")

async def models_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    files = [f for f in os.listdir(config.MODELS_DIR) if f.endswith('.h5')]
    resp = "<b>Dostępne modele:</b>\n" + "\n".join([f"{f}" for f in files]) if files else " Brak modeli."
    await update.message.reply_text(resp, parse_mode="HTML")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_text = (
        "**Dostępne komendy systemu NLP:**\n\n"
        "**Podstawy i statystyki:**\n"
        "`/task <tekst>` - Przetwarzanie (tokeny, lematy)\n"
        "`/stats <tekst>` - Wykresy i chmura słów\n"
        "`/classify <tekst>` - Szybka klasyfikacja ML\n\n"
        "**Głębokie sieci i Transformery:**\n"
        "`/sentiment method=<metoda> text=\"<tekst>\" [dataset=<baza>]` - Analiza sentymentu\n"
        "`/train <simplernn|lstm|gru> <dataset>` - Trenowanie modelu\n"
        "`/compare dataset=<baza> methods=<met1,met2>` - Zaawansowana ewaluacja i macierze pomyłek\n"
        "`/add_sentiment \"<tekst>\" \"<etykieta>\"` - Rozbudowa własnej bazy\n"
        "`/models` - Lista wgranych modeli\n"
        "`/help` - Ten komunikat"
    )
    await update.message.reply_text(help_text, parse_mode="Markdown")