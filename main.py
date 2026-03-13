import os
import logging
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler

# Importujemy nasze komendy z nowego modułu
from bot.handlers import (
    start_command, 
    task_command, 
    full_pipeline_command, 
    classifier_command, 
    stats_command
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", 
    level=logging.INFO
)

logging.getLogger("httpx").setLevel(logging.WARNING)

load_dotenv()
TOKEN = os.getenv("TELEGRAM_TOKEN")

def main():
    if not TOKEN:
        print("BŁĄD: Nie znaleziono tokena! Upewnij się, że masz plik .env z TELEGRAM_TOKEN.")
        return

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("task", task_command))
    app.add_handler(CommandHandler("full_pipeline", full_pipeline_command))
    app.add_handler(CommandHandler("classifier", classifier_command))
    app.add_handler(CommandHandler("stats", stats_command))

    print("Bot uruchamia się pomyślnie. Naciśnij Ctrl+C, aby zatrzymać.")
    app.run_polling()

if __name__ == "__main__":
    main()