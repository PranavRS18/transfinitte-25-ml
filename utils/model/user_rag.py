import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests

# Base folder to store user indices
USER_INDEX_DIR = os.path.join(os.path.dirname(__file__), "user_indices")
os.makedirs(USER_INDEX_DIR, exist_ok=True)

# Helper function to get file path
def get_index_path(user_id):
    return os.path.join(USER_INDEX_DIR, f"{user_id}_index.faiss")

def get_doc_path(user_id):
    return os.path.join(USER_INDEX_DIR, f"{user_id}_docs.json")

class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def embed_documents(self, documents):
        self.documents = documents
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self.index

    def save_index(self, user_id):
        if self.index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, get_index_path(user_id))
        # Save docs as JSON
        with open(get_doc_path(user_id), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)

    def load_index(self, user_id):
        index_path = get_index_path(user_id)
        doc_path = get_doc_path(user_id)
        if not os.path.exists(index_path) or not os.path.exists(doc_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(doc_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        return True

    def retrieve(self, query, top_k=2):
        if self.index is None or not self.documents:
            raise ValueError("No index loaded.")
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        system_prompt = "You are a helpful assistant. Use the following context to answer the question:\n\n"
        context = "\n\n".join(retrieved_docs)
        return {"query": query, "retrieved_docs": retrieved_docs, "system_prompt": system_prompt, "context": context}

# Functions for Node.js routes
def add_documents(user_id, documents):
    retr = Retriever()
    # Load existing index if exists
    retr.load_index(user_id)
    # Combine old and new documents
    retr.documents.extend(documents)
    retr.embed_documents(retr.documents)
    retr.save_index(user_id)
    return {"status": "success", "message": f"{len(documents)} docs added for {user_id}"}

def rag_query(user_id, query, top_k=2):
    retr = Retriever()
    if not retr.load_index(user_id):
        return {"error": f"No documents found for user {user_id}"}
    result = retr.retrieve(query, top_k)
    # Send to Ollama
    data = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": result["system_prompt"]},
            {"role": "user", "content": query}
        ],
        "max_tokens": 200
    }
    try:
        response = requests.post("http://localhost:11434/v1/chat/completions", json=data, timeout=30)
        answer = response.json()["choices"][0]["message"]["content"]
        return {"query": query, "answer": answer, "retrieved_docs": result["retrieved_docs"]}
    except Exception as e:
        return {"error": str(e)}


