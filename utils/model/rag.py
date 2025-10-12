import sys
import requests
import json
from utils.model.init import Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv import load_dotenv
from pinecone import Pinecone
# --- Initialize retriever (but don't embed anything yet) ---
retr = Retriever()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

def embed_user_docs(text):
    """Embed documents provided later by the user."""
    print("Embedding user documents...", text)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=150,
        chunk_overlap=50
    )
    chunks = splitter.split_text(text)
    retr.embed_documents(chunks)

def rag_answer(user_prompt, model = "ollama"):
    """Perform retrieval + generation using the specified model."""
    # Retrieve relevant documents
    result = retr.retrieve(user_prompt, top_k=2,backend="pinecone")
    system_prompt = result.get("system_prompt", "You are a helpful assistant.")

    # ---- Choose Backend ----
    if model == "ollama":
        # Ollama API (local)
        data = {
            "model": "llama3.2:1b",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_tokens": 200
        }
        url = OLLAMA_API_URL

    elif model == "gemini":
        # Gemini API (Google Generative Language)
        url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=YOUR_API_KEY"
        data = {
            "contents": [
                {
                    "parts": [
                        {"text": f"System: {system_prompt}\nUser: {user_prompt}"}
                    ]
                }
            ]
        }
    else:
        raise ValueError("Unsupported model. Use 'ollama' or 'gemini'.")

    # ---- Send request ----
    try:
        response = requests.post(url, json = data, timeout = 40)
        response.raise_for_status()
        resp_json = response.json()

        # Handle Ollama response
        if model == "ollama":
            return resp_json["choices"][0]["message"]["content"]

        # Handle Gemini response
        elif model == "gemini":
            return resp_json["candidates"][0]["content"]["parts"][0]["text"]

    except requests.exceptions.RequestException as e:
        return f"Error: Could not reach {model} API. Details: {e}"
    except Exception as e:
        return f"Unexpected error: {e}"

# --- Main CLI interface ---
if __name__ == "__main__":
    user_input = sys.stdin.read().strip()
    if not user_input:
        print("Error: No prompt received.")
        sys.exit(1)

    # Default: use Ollama locally
    print(rag_answer(user_input, model="ollama"))
