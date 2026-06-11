import requests
from nlp.tools import AVAILABLE_TOOLS_SCHEMA, TOOLS_MAPPING

def run_agent(user_query: str) -> str:
    """Main function of the autonomous Agent AI with visual support."""
    
    # 1. Agresywny System Prompt nadpisujący wbudowane blokady modelu (Refusal Bias)
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI with access to the internet, a calculator, and a local vision module. "
                "You MUST strictly use the provided tools to answer. "
                "When the user mentions 'image' or 'photo', ALWAYS assume they mean the uploaded file "
                "and immediately use the 'analyze_image' tool with the correct file path. "
                "CRITICAL INSTRUCTION: When you receive the image description from the tool, "
                "DO NOT invent metadata (like pixels, format) or add scenic details not present in the tool's output. "
                "Simply present the tool's findings to the user."
            )
        },
        {"role": "user", "content": user_query}
    ]

    payload = {
        "model": "llama3.2",
        "messages": messages,
        "tools": AVAILABLE_TOOLS_SCHEMA,
        "stream": False
    }

    try:
        response = requests.post("http://localhost:11434/api/chat", json=payload, timeout=120).json()
        message = response.get("message", {})

        if "tool_calls" in message and message["tool_calls"]:
            messages.append(message)
            used_tools_log = []

            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = tool_call["function"]["arguments"]

                print(f"[Agent] Wywołuję narzędzie: {func_name} z argumentami: {func_args}")
                # Zapisujemy log do wyświetlenia użytkownikowi na Telegramie
                used_tools_log.append(f"*(Użyto narzędzia: {func_name})*")

                if func_name in TOOLS_MAPPING:
                    func = TOOLS_MAPPING[func_name]
                    try:
                        tool_result = func(**func_args)
                        print(f" [Wynik narzędzia]: {tool_result}") 
                    except Exception as e:
                        tool_result = f"Error executing tool: {e}"
                else:
                    tool_result = "Error: Tool not found."

                messages.append({
                    "role": "tool",
                    "content": str(tool_result),
                    "name": func_name
                })

            second_payload = {
                "model": "llama3.2",
                "messages": messages,
                "stream": False
            }

            print("[Agent] Analizuję wyniki narzędzi i generuję odpowiedź...")
            second_response = requests.post("http://localhost:11434/api/chat", json=second_payload, timeout=120).json()
            final_answer = second_response.get("message", {}).get("content", "Błąd przetwarzania.")

            # Doklejamy logi na samej górze odpowiedzi
            return "\n".join(used_tools_log) + "\n\n" + final_answer

        else:
            print("[Agent] Odpowiadam bezpośrednio z pamięci modelu.")
            return message.get("content", "Brak odpowiedzi.")

    except requests.exceptions.RequestException as e:
        return f"Błąd komunikacji z serwerem Ollama: {e}"