import os
import sys
import requests
from google import genai

# --- Gemini setup ---
api_key = "AIzaSyDgxIc3UsBdft3UjGTGBnuqex5epK9APcc"
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set.", file=sys.stderr)
    sys.exit(1)

client = genai.Client(api_key=api_key)


# --- Ollama summarization ---
def summarise_ollama(text, model_name="llama3.2:1b"):
    url = "http://localhost:11434/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"Summarize the text: {text}"}
        ],
        "max_tokens": 150
    }
    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"Ollama error ({model_name}): {e}", file=sys.stderr)
        return None


# --- Gemini summarization ---
def summarise_gemini(text, model_name="gemini-2.5-flash"):
    try:
        prompt = f"Summarize the following text:\n{text}"
        response = client.models.generate_content(model=model_name, contents=prompt)
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"Gemini error ({model_name}): {e}", file=sys.stderr)
        return None


# --- Unified summarization ---
def summarise_any(text, provider="ollama", model_name=None):
    if provider.lower() == "ollama":
        return summarise_ollama(text, model_name)
    elif provider.lower() == "gemini":
        return summarise_gemini(text, model_name)
    else:
        print(f"Unknown provider: {provider}", file=sys.stderr)
        return None


# --- Entry point (called by Node.js) ---
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python summarise.py <provider> <model_name> <text>", file=sys.stderr)
        sys.exit(1)

    provider = sys.argv[1]
    model_name = sys.argv[2]
    text = " ".join(sys.argv[3:])  # join all remaining args into one text string

    summary = summarise_any(text, provider=provider, model_name=model_name)
    if summary:
        print(summary)
    else:
        print("Error: Summarization failed", file=sys.stderr)
        sys.exit(1)
