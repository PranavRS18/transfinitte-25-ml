import os
import json
import faiss
import requests
import sys
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
load_dotenv()

OLLAMA_API_URL = os.getenv("OLLAMA_API_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from sentence_transformers import SentenceTransformer

# Directory for storing FAISS indices per user
USER_INDEX_DIR = os.path.join(os.path.dirname(__file__), "user_indices")
os.makedirs(USER_INDEX_DIR, exist_ok=True)

# --- Helper path builders ---
def get_index_path(userId):
    return os.path.join(USER_INDEX_DIR, f"{userId}_index.faiss")

def get_doc_path(userId):
    return os.path.join(USER_INDEX_DIR, f"{userId}_docs.json")


class Retriever:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []
        self.pc = Pinecone(
            api_key=os.getenv("PINECONE_API_KEY"),
            environment="us-east-1"
        )

# Index name
        index_name = os.getenv("PINECONE_INDEX_NAME")

        #
        # Initialize the Pinecone index
        self.pinecone_index = self.pc.Index(index_name)

    # --- Embedding and persistence ---
    def embed_documents(self, documents):
        """Embed and index a list of documents."""
        self.documents = documents
        embeddings = self.model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        vectors = []
        for i, emb in enumerate(embeddings):
            vectors.append({
                "id": f"{userId}-{i}",
                "values": emb.tolist(),
                "metadata": {"text": documents[i], "user": userId},
            })
        self.pinecone_index.upsert(vectors=vectors)
        return self.index

    def save_index(self, userId):
        """Save FAISS index + docs to disk."""
        if self.index is None:
            raise ValueError("Index not built.")
        faiss.write_index(self.index, get_index_path(userId))
        with open(get_doc_path(userId), "w", encoding="utf-8") as f:
            json.dump(self.documents, f)

    def load_index(self, userId):
        """Load FAISS index + docs from disk."""
        index_path = get_index_path(userId)
        doc_path = get_doc_path(userId)
        if not os.path.exists(index_path) or not os.path.exists(doc_path):
            return False
        self.index = faiss.read_index(index_path)
        with open(doc_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)
        return True

    # --- Retrieval ---
    def retrieve(self, query, top_k=2, backend="pinecone"):
        """
        Retrieve top-k docs given a query from either Pinecone or FAISS.
        
        Args:
            query (str): The input query.
            top_k (int): Number of top documents to retrieve.
            backend (str): "pinecone" or "faiss".
        
        Returns:
            dict: Contains query, retrieved_docs, system_prompt, and context.
        """
        query_vector = self.model.encode([query], convert_to_numpy=True)[0]

        retrieved_docs = []

        if backend.lower() == "pinecone":
            if self.pinecone_index is None:
                raise ValueError("Pinecone index not initialized.")
            response = self.pinecone_index.query(
                vector=query_vector.tolist(),
                top_k=top_k,
                include_metadata=True
            )
            retrieved_docs = [match['metadata']['text'] for match in response['matches']]

        elif backend.lower() == "faiss":
            if self.index is None or not self.documents:
                raise ValueError("FAISS index or documents not loaded.")
            distances, indices = self.index.search(query_vector.reshape(1, -1), top_k)
            retrieved_docs = [self.documents[i] for i in indices[0]]

        else:
            raise ValueError("Invalid backend. Choose 'pinecone' or 'faiss'.")

        system_prompt = "You are a helpful assistant. Use the following context to answer accurately: (Limit your answer to 10 words)\n\n"
        context = "\n\n".join(retrieved_docs)

        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "system_prompt": system_prompt,
            "context": context,
        }



# --- Document management ---
def add_documents(userId, documents):
    """Add or update documents for a user."""
    retr = Retriever()
    retr.load_index(userId)
    retr.documents.extend(documents)
    retr.embed_documents(retr.documents)
    retr.save_index(userId)
    return {"status": "success", "message": f"{len(documents)} documents added for {userId}"}


# --- Model-Agnostic RAG Query ---
def rag_query(userId, query, model_name="llama3.2:1b", top_k=2,backend="pinecone"):
    retr = Retriever()
    if not retr.load_index(userId):
        return {"error": f"No documents found for user {userId}"}

    result = retr.retrieve(query, top_k,backend=backend)
    context = result["context"]
    system_prompt = result["system_prompt"]

    # --- Determine provider ---
    if "gemini" in model_name.lower():
        # --- Gemini API ---
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={GEMINI_API_KEY}"
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
        userId = sys.argv[2]
        raw_docs = sys.argv[3]

        try:
            documents = json.loads(raw_docs)
            if isinstance(documents, str):
                documents = [documents]
        except json.JSONDecodeError:
            documents = [raw_docs]

        result = add_documents(userId, documents)
        print(json.dumps(result))

    elif action == "query":
        userId = sys.argv[2]
        query = sys.argv[3]
        model_name = sys.argv[4]
        db = sys.argv[5] if len(sys.argv) > 5 else "pinecone"

        result = rag_query(userId, query, model_name = model_name, backend = db)
        print(json.dumps(result))