import sys
from utils.model.init import Retriever
import requests
import json

# Create retriever and embed documents once
retr = Retriever()
docs = [
    "Python is a programming language.",
    "FastAPI is a modern web framework for building APIs.",
    "The earth revolves around the sun."
]
retr.embed_documents(docs)

def rag_answer(user_prompt):
    # Retrieve relevant documents
    result = retr.retrieve(user_prompt, top_k=2)
    system_prompt = result["system_prompt"]

    data = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 200
    }

    response = requests.post(
        "http://localhost:11434/v1/chat/completions",
        json=data,
        timeout=30
    )
    return response.json()["choices"][0]["message"]["content"]

# Read user prompt from stdin (sent by Node.js)
if __name__ == "__main__":
    prompt = sys.stdin.read().strip()
    if not prompt:
        print("Error: No prompt received")
        sys.exit(1)

    try:
        answer = rag_answer(prompt)
        print(answer)
    except requests.exceptions.RequestException as e:
        print(f"Error connecting to Ollama: {e}")
        sys.exit(1)
