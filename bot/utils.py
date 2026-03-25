import os
import json
import asyncio
import aiofiles

# Ścieżka względem głównego folderu uruchomieniowego (main.py)
DATA_FILE = "data/sentences.json"

# Globalna blokada chroniąca plik przed jednoczesnym zapisem z wielu wątków
file_lock = asyncio.Lock()

async def save_sentence_to_json(text: str, text_class: str):
    """Asynchronicznie zapisuje nowy rekord do pliku JSON."""
    new_record = {"text": text, "class": text_class}
    
    # Czekamy w kolejce na dostęp do pliku
    async with file_lock:
        if os.path.exists(DATA_FILE):
            # Asynchroniczny odczyt
            async with aiofiles.open(DATA_FILE, mode="r", encoding="utf-8") as file:
                try:
                    content = await file.read()
                    data = json.loads(content) if content else []
                except json.JSONDecodeError:
                    data = [] 
        else:
            os.makedirs("data", exist_ok=True)
            data = []
            
        data.append(new_record)
        
        # Asynchroniczny zapis
        async with aiofiles.open(DATA_FILE, mode="w", encoding="utf-8") as file:
            await file.write(json.dumps(data, ensure_ascii=False, indent=2))