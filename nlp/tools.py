import requests
import json
import base64
from duckduckgo_search import DDGS
import base64

# 1. IMPLEMENTACJA FUNKCJI (Narzędzia Pythona)

def get_weather(city: str) -> str:
    """Return current weather for a given city."""
    try:
        # 1. Geocoding: Zamiana nazwy miasta przekazanej przez LLM na współrzędne
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}&count=1"
        geo_res = requests.get(geo_url, timeout=5).json()
        
        if "results" not in geo_res:
            return f"Błąd: Nie znaleziono współrzędnych dla miasta {city}."
            
        lat = geo_res["results"][0]["latitude"]
        lon = geo_res["results"][0]["longitude"]
        
        # 2. Pobranie pogody dla odzyskanych, dynamicznych współrzędnych
        weather_url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        weather_res = requests.get(weather_url, timeout=5).json()
        weather = weather_res["current_weather"]
        
        return f"Current weather in {city}: {weather['temperature']}°C, wind {weather['windspeed']} km/h"
    except Exception as e:
        return f"Error fetching weather: {e}"

def simple_calculator(expression: str) -> str:
    """Evaluate simple math expression."""
    try:
        result = eval(expression, {"__builtins__": None}, {})
        return f"Result of {expression} is {result}"
    except Exception:
        return "Error: Invalid math expression."

def local_knowledge(query: str) -> str:
    """Search in local knowledge base."""
    db = {
        "prowadzący": "Zajęcia z przedmiotu NLP prowadzi doświadczony inżynier AI.",
        "zaliczenie": "Warunkiem zaliczenia jest oddanie w pełni działającego bota na platformę Telegram.",
        "temat": "Laboratorium 5 dotyczy Function Calling i modeli Agentowych."
    }
    for key, value in db.items():
        if key in query.lower():
            return value
    return "No relevant information found in the local database."

def web_search(query: str) -> str:
    """Search the internet for up-to-date information."""
    try:
        results = DDGS().text(query, max_results=3)
        
        # 1. Wyszukiwarka zwróciła pustą listę wyników, co może oznaczać tymczasową blokadę anty-botową
        if not results:
            return "Błąd: Wyszukiwarka nie zwróciła żadnych wyników. Możliwa tymczasowa blokada anty-botowa."
            
        formatted_results = "\n".join([f"Source: {r['title']}\nContent: {r['body']}" for r in results])
        return formatted_results
        
    except Exception as e:
        error_msg = str(e).lower()
        # 2. Twarda blokada (Rate Limit) rzucona jako wyjątek z biblioteki
        if "ratelimit" in error_msg or "429" in error_msg:
            return "Błąd: Serwer DuckDuckGo zablokował adres IP za zbyt dużo zapytań (Rate Limit 429). Odczekaj chwilę."
            
        return f"Błąd narzędzia web_search: {e}"

def analyze_image(image_path: str, question: str) -> str:
    """Analyze a local image using the LLaVA vision model."""
    try:
        # 1. Uproszczony prompt: małe modele wizyjne lepiej reagują na proste polecenia
        if not question or question.strip() == "":
            question = "What is in this image? Describe the main subject."
            
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
        
        payload = {
            "model": "llava",
            "prompt": question,
            "images": [encoded_string],
            "stream": False
        }
        
        response = requests.post("http://localhost:11434/api/generate", json=payload, timeout=300)
        
        # 2. Wyciągnięcie surowego wyniku z Ollamy
        raw_output = response.json().get("response", "ERROR_EMPTY_RESPONSE")
        
        # 3. TELEMETRIA: Wymuszamy wypisanie surowego wyniku w terminalu dla celów debugowania
        print(f"\n[DEBUG LLaVA RAW]: {raw_output}\n")
        
        return raw_output
        
    except Exception as e:
        return f"CRITICAL ERROR: {e}. DO NOT invent a description."

# 2. SCHEMATY JSON DLA MODELU LLM

weather_tool = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "The name of the city, e.g. Warsaw, Paris"}
            },
            "required": ["city"]
        }
    }
}

calculator_tool = {
    "type": "function",
    "function": {
        "name": "simple_calculator",
        "description": "Evaluate basic mathematical expressions (addition, subtraction, multiplication, division)",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "The mathematical expression, e.g. '2 + 2' or '10 * 5'"}
            },
            "required": ["expression"]
        }
    }
}

knowledge_tool = {
    "type": "function",
    "function": {
        "name": "local_knowledge",
        "description": "Search the local knowledge base for information about the university, classes, or course requirements.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search term or question."}
            },
            "required": ["query"]
        }
    }
}

web_search_tool = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the internet for current events, news, or up-to-date factual information.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query to type into the search engine."}
            },
            "required": ["query"]
        }
    }
}

vision_tool = {
    "type": "function",
    "function": {
        "name": "analyze_image",
        "description": "Analyze an image file stored on the local disk.",
        "parameters": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string", "description": "The absolute or relative file path to the image."},
                "question": {"type": "string", "description": "The specific question to ask about the image's content."}
            },
            "required": ["image_path", "question"]
        }
    }
}

AVAILABLE_TOOLS_SCHEMA = [
    weather_tool, 
    calculator_tool, 
    knowledge_tool, 
    web_search_tool, 
    vision_tool
]

TOOLS_MAPPING = {
    "get_weather": get_weather,
    "simple_calculator": simple_calculator,
    "local_knowledge": local_knowledge,
    "web_search": web_search,
    "analyze_image": analyze_image
}