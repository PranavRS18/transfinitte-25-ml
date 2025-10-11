import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
import json


class Retriever:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        # Load embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def embed_documents(self, documents):
        """Encode and store document embeddings in FAISS index."""
        self.documents = documents
        embeddings = self.embedding_model.encode(documents, convert_to_numpy=True, show_progress_bar=False)
        
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def retrieve(self, query, top_k=2):
        """Retrieve top-k similar documents for the query."""
        if self.index is None or not self.documents:
            raise ValueError("Index not built. Call embed_documents(documents) first.")
        
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_vector, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]
        
        # Optional: create system prompt context
        system_prompt = "You are a helpful assistant. Use the following context to answer the question:\n\n"
        context = "\n\n".join(retrieved_docs)
        
        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "system_prompt": system_prompt,
            "context": context,
            "distances": distances.tolist()
        }


# Example Usage
if __name__ == "__main__":
    docs = [
        "Python is a programming language.",
        "FastAPI is a modern web framework for building APIs.",
        "The earth revolves around the sun."
    ]

    retriever = Retriever()
    retriever.embed_documents(docs)
    result = retriever.retrieve("What is FastAPI?", top_k=2)

    print("Retrieved Documents:", result["retrieved_docs"])
