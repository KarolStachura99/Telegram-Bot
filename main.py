import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

# 1. Konfiguracja logowania 
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 2. Wczytanie bezpiecznego tokena z pliku .env
load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

# --- DEFINICJE KOMEND ---

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Obsługa powitania."""
    await update.message.reply_text("Cześć! Jestem Botem NLP. Gotowy do pracy!")

async def task_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atrapa dla komendy /task"""
    await update.message.reply_text("To jest odpowiedź testowa z komendy /task. Tu w przyszłości będzie pojedyncza logika NLP i zapis do JSON.")

async def full_pipeline_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atrapa dla komendy /full_pipeline"""
    await update.message.reply_text("To jest odpowiedź testowa z komendy /full_pipeline. Tutaj przeprowadzimy pełną analizę tekstu!")

async def classifier_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atrapa dla komendy /classifier"""
    await update.message.reply_text("To jest odpowiedź testowa z komendy /classifier. Tu podepniemy model scikit-learn.")

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Atrapa dla komendy /stats"""
    await update.message.reply_text("To jest odpowiedź testowa z komendy /stats. Tu wygenerujemy globalne wykresy i statystyki korpusu.")

# --- GŁÓWNA FUNKCJA URUCHAMIAJĄCA BOTA ---

def main():
    if not TOKEN:
        print("BŁĄD: Nie znaleziono tokena! Upewnij się, że masz plik .env z TELEGRAM_TOKEN.")
        return

    # Budowanie aplikacji bota
    app = Application.builder().token(TOKEN).build()

    # Rejestrowanie komend, żeby bot wiedział, jak na nie reagować
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("task", task_command))
    app.add_handler(CommandHandler("full_pipeline", full_pipeline_command))
    app.add_handler(CommandHandler("classifier", classifier_command))
    app.add_handler(CommandHandler("stats", stats_command))

    # Uruchomienie nasłuchiwania w pętli
    print("Bot uruchamia się pomyślnie. Naciśnij Ctrl+C, aby zatrzymać.")
    app.run_polling()

if __name__ == "__main__":
    main()