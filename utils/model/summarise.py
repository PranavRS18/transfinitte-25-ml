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

    response = requests.post(url, headers = {"Content-Type": "application/json"}, data = json.dumps(data))
    
    return response.json()["choices"][0]["message"]["content"]
