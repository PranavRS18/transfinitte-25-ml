import sys
import requests
import json

def summarise(text):
    url = "http://localhost:11434/v1/chat/completions"
    data = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": "You are a summarization assistant."},
            {"role": "user", "content": f"Summarize the text: {text}"}
        ],
        "max_tokens": 150
    }

    try:
        response = requests.post(url, json=data, timeout=30)
        response.raise_for_status()
        summary = response.json()["choices"][0]["message"]["content"].strip()
        return summary
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        return None
    except KeyError:
        print("Unexpected response format:", response.text)
        return None

if __name__ == "__main__":
    # Read text from stdin (sent from Node.js)
    text = sys.stdin.read().strip()
    if not text:
        print("Error: No text received")
        sys.exit(1)
    
    summary = summarise(text)
    if summary:
        print(summary)
    else:
        print("Error: Summarization failed")
