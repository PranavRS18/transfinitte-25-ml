import os
import json
import faiss
import requests
import sys


from sentence_transformers import SentenceTransformer

# Directory for storing FAISS indices per user
USER_INDEX_DIR = os.path.join(os.path.dirname(__file__), "user_indices")
os.makedirs(USER_INDEX_DIR, exist_ok=True)

# --- Helper path builders ---
def get_index_path(user_id):
    return os.path.join(USER_INDEX_DIR, f"{user_id}_index.faiss")

def get_doc_path(user_id):
    return os.path.join(USER_INDEX_DIR, f"{user_id}_docs.json")


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    # --- Embedding and persistence ---
    def embed_documents(self, documents):
        """Embed and index a list of documents."""
        self.documents = documents
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        return self.index

    def save_index(self, user_id):
        """Save FAISS index + docs to disk."""
        if self.index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, get_index_path(user_id))
        with open(get_doc_path(user_id), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)

    def load_index(self, user_id):
        """Load FAISS index + docs from disk."""
        index_path = get_index_path(user_id)
        doc_path = get_doc_path(user_id)
        if not os.path.exists(index_path) or not os.path.exists(doc_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(doc_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        return True

    # --- Retrieval ---
    def retrieve(self, query, top_k=2):
        """Retrieve top-k docs given a query."""
        if self.index is None or not self.documents:
            raise ValueError("No index loaded.")
        query_vector = self.model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        system_prompt = "You are a helpful assistant. Use the following context to answer accurately:\n\n"
        context = "\n\n".join(retrieved_docs)
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "system_prompt": system_prompt,
            "context": context,
        }


# --- Document management ---
def add_documents(user_id, documents):
    """Add or update documents for a user."""
    retr = Retriever()
    retr.load_index(user_id)
    retr.documents.extend(documents)
    retr.embed_documents(retr.documents)
    retr.save_index(user_id)
    return {"status": "success", "message": f"{len(documents)} documents added for {user_id}"}


# --- Model-agnostic RAG Query ---
def rag_query(user_id, query, model_name="llama3.2:1b", top_k=2):
    retr = Retriever()
    if not retr.load_index(user_id):
        return {"error": f"No documents found for user {user_id}"}

    result = retr.retrieve(query, top_k)
    context = result["context"]
    system_prompt = result["system_prompt"]

    # --- Determine provider ---
    if "gemini" in model_name.lower():
        # --- Gemini API ---
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key=AIzaSyDgxIc3UsBdft3UjGTGBnuqex5epK9APcc"
        data = {
            "contents": [
                {"parts": [{"text": f"{system_prompt}\nContext:\n{context}\n\nQuestion: {query}"}]}
            ]
        }
    else:
        # --- Ollama local model ---
        url = OLLAMA_API_URL
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt + context},
                {"role": "user", "content": query},
            ],
            "max_tokens": 200,
        }

    # --- Make API call ---
    try:
        resp = requests.post(url, json=data, timeout=40)
        resp.raise_for_status()
        response = resp.json()

        if "gemini" in model_name.lower():
            answer = response["candidates"][0]["content"]["parts"][0]["text"]
        else:
            answer = response["choices"][0]["message"]["content"]

        return {
            "query": query,
            "answer": answer,
            "retrieved_docs": result["retrieved_docs"],
            "model_name": model_name,
        }

    except Exception as e:
        return {"error": str(e)}
if __name__ == "__main__":
    action = sys.argv[1]  # "add" or "query"

    if action == "add":
        user_id = sys.argv[2]
        documents = json.loads(sys.argv[3])
        result = add_documents(user_id, documents)
        print(json.dumps(result))

    elif action == "query":
        user_id = sys.argv[2]
        query = sys.argv[3]
        model_name = sys.argv[4]
        result = rag_query(user_id, query, model_name=model_name)
        print(json.dumps(result))