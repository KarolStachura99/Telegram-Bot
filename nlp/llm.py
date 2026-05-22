import requests
import time

def summarize_text_with_ollama(text: str, summary_type: str = "abstractive", length: str = "medium") -> tuple[str, float]:
    """
    Łączy się z lokalnym API Ollamy w celu wygenerowania streszczenia.
    Zwraca krotkę: (wygenerowany_tekst, czas_generowania).
    """
    url = "http://localhost:11434/api/generate"
    
    # Budowanie promptu sterującego
    prompt = (
        "You are a professional AI assistant. Analyze and summarize the provided text. "
        "YOUR RESPONSE MUST BE ENTIRELY IN POLISH. "
        "CRITICAL RULE: You must rely EXCLUSIVELY on the provided text. "
        "DO NOT invent, guess, or add any external information, facts, or historical events not explicitly mentioned in the source.\n"
    )

    if summary_type == "extractive":
        prompt += "Extract and quote only the most important key sentences in their original form.\n"
    elif summary_type == "bullets":
        prompt += "Provide the summary as short, concise bullet points.\n"
    else:  # abstractive
        prompt += "Write a coherent, abstractive summary in your own words.\n"
        
    if length == "short":
        prompt += "The summary must be very short (maximum 1-2 sentences).\n"
    elif length == "long":
        prompt += "The summary must be long, detailed, and comprehensive.\n"
    else:
        prompt += "The summary should be of medium length.\n"
        
    prompt += f"\nTEXT TO SUMMARIZE:\n{text}"
    
    payload = {
        "model": "llama3.2",  # Upewnij się, że ten model został pobrany (ollama run llama3.2)
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    
    try:
        # Wysyłamy zapytanie do lokalnego serwera Ollamy (ustawiamy timeout na wypadek długich tekstów)
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        
        result = response.json()
        generation_time = round(time.time() - start_time, 2)
        
        return result.get("response", "Brak odpowiedzi od modelu."), generation_time
        
    except requests.exceptions.RequestException as e:
        return f"Błąd połączenia z lokalnym serwerem Ollama. Upewnij się, że aplikacja jest włączona. Szczegóły: {e}", 0.0