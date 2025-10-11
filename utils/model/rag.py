from utils.model.init import Retriever
import requests
import json

# 1️⃣ Create retriever instance
retr = Retriever()

# 2️⃣ Embed your documents
docs = [
    "Python is a programming language.",
    "FastAPI is a modern web framework for building APIs.",
    "The earth revolves around the sun."
]
retr.embed_documents(docs)

# 3️⃣ RAG answer function
def rag_answer(user_prompt):
    # Use retriever instance
    result = retr.retrieve(user_prompt, top_k=2)
    system_prompt = result["system_prompt"]
    context = result["context"]

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
        json=data,  # simpler than data=json.dumps()
        timeout=30
    )

    return response.json()["choices"][0]["message"]["content"]


# 4️⃣ Test
if __name__ == "__main__":
    prompt = "Explain what FastAPI is in simple words."
    try:
        answer = rag_answer(prompt)
        print("✅ LLM Answer:\n", answer)
    except requests.exceptions.ConnectionError:
        print("❌ Ollama server not running! Run:\nollama run llama3.2:1b")
