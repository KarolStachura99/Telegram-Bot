import os

# ==============================================================================
# --- BRUTALNA BLOKADA WIELOWĄTKOWOŚCI C++ I TENSORFLOW (OCHRONA WINDOWSA) ---
# MUSI BYĆ NA SAMEJ GÓRZE, PRZED JAKIMIKOLWIEK INNYMI IMPORTAMI!
# ==============================================================================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'


import logging
from dotenv import load_dotenv
from telegram.ext import Application, CommandHandler

from bot.commands import(
    train_command,
    start_command,
    task_command,
    full_pipeline_command,
    classifier_command,
    stats_command,
    classify_command,
    sentiment_command,
    add_sentiment_command,
    models_command,
    compare_command,
    help_command
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
        print("BŁĄD: Nie znaleziono tokena!")
        return

    app = Application.builder().token(TOKEN).build()

    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("task", task_command))
    app.add_handler(CommandHandler("full_pipeline", full_pipeline_command))
    app.add_handler(CommandHandler("classifier", classifier_command))
    app.add_handler(CommandHandler("stats", stats_command))
    app.add_handler(CommandHandler("classify", classify_command))
    app.add_handler(CommandHandler("sentiment", sentiment_command))
    app.add_handler(CommandHandler("train", train_command))
    app.add_handler(CommandHandler("add_sentiment", add_sentiment_command))
    app.add_handler(CommandHandler("models", models_command))
    app.add_handler(CommandHandler("compare", compare_command))
    app.add_handler(CommandHandler("help", help_command))

    print("Bot uruchamia się pomyślnie. Naciśnij Ctrl+C, aby zatrzymać.")
    app.run_polling()


if __name__ == "__main__":
    main()