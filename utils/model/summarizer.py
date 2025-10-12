import os
import sys
import requests
from google import genai

from dotenv import load_dotenv
load_dotenv()

# --- Gemini Setup ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY environment variable not set.", file = sys.stderr)
    sys.exit(1)

client = genai.Client(api_key = GEMINI_API_KEY)

# --- Ollama Setup ---
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
if not OLLAMA_API_URL:
    print("Error: OLLAMA_API_URL environment variable not set.", file = sys.stderr)
    sys.exit(1)

# --- Ollama Summarization ---
def summarise_ollama(text, model_name = "llama3.2:1b"):
    url = OLLAMA_API_URL
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"Summarize the text: {text}"}
        ],
        "max_tokens": 150
    }
    try:
        response = requests.post(url, json = data, timeout = 30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama error ({model_name}): {e}", file=sys.stderr)
        return None


# --- Gemini Summarization ---
def summarise_gemini(text, model_name = "gemini-2.5-flash"):
    try:
        prompt = f"Summarize the following text: \n{text}"
        response = client.models.generate_content(model = model_name, contents = prompt)
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"Gemini error ({model_name}): {e}", file=sys.stderr)
        return None


# --- Fallback Summarisation ---
def summarise_any(text, provider = "ollama", model_name=None):
    if provider.lower() == "ollama":
        return summarise_ollama(text, model_name)
    elif provider.lower() == "gemini":
        return summarise_gemini(text, model_name)
    else:
        print(f"Unknown provider: {provider}", file = sys.stderr)
        return None


# --- Entry Point (called by Node.js) ---
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python summarise.py <provider> <model_name> <text>", file = sys.stderr)
        sys.exit(1)

    provider = sys.argv[1]
    model_name = sys.argv[2]
    text = sys.stdin.read().strip()

    summary = summarise_any(text, provider = provider, model_name = model_name)

    if summary:
        print(summary)
    else:
        print("Error: Summarization failed", file = sys.stderr)
        sys.exit(1)
