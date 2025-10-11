import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class Retriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the Retriever with a sentence-transformers embedding model.
        """
        self.embedding_model = SentenceTransformer(model_name)
        self.index = None
        self.documents = []

    def embed_documents(self, documents: list):
        """
        Encode documents and store them in a FAISS index.

        Args:
            documents (list[str]): List of document strings to embed.
        """
        if not documents:
            raise ValueError("No documents provided to embed.")

        self.documents = documents
        embeddings = self.embedding_model.encode(
            documents, convert_to_numpy=True, show_progress_bar=True
        )

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def add_documents(self, new_documents: list):
        """
        Add new documents to the existing index.

        Args:
            new_documents (list[str]): List of new document strings to embed and add.
        """
        if not new_documents:
            return

        combined_docs = self.documents + new_documents
        self.embed_documents(combined_docs)

    def retrieve(self, query: str, top_k: int = 2) -> dict:
        """
        Retrieve top-K most similar documents for a query.

        Args:
            query (str): The query string.
            top_k (int): Number of top documents to return.

        Returns:
            dict: Contains query, retrieved documents, context, and distances.
        """
        if self.index is None or not self.documents:
            raise ValueError("Index not built. Call embed_documents() first.")

        # Encode query
        query_vector = self.embedding_model.encode([query], convert_to_numpy=True)

        # Search FAISS index
        distances, indices = self.index.search(query_vector, top_k)
        retrieved_docs = [self.documents[i] for i in indices[0]]

        # Create context for system prompt
        system_prompt = (
            "You are a helpful assistant. Use the following context to answer the question:\n\n"
        )
        context = "\n\n".join(retrieved_docs)

        return {
            "query": query,
            "retrieved_docs": retrieved_docs,
            "system_prompt": system_prompt,
            "context": context,
            "distances": distances.tolist(),
        }


# ----------------- Example Usage -----------------
if __name__ == "__main__":
    retriever = Retriever()

    # Dynamically add documents later instead of hardcoding
    # retriever.embed_documents(["Your documents here..."])

    # Example query (will fail if no documents are added)
    try:
        result = retriever.retrieve("What is FastAPI?", top_k=2)
        print("Retrieved Documents:", result["retrieved_docs"])
    except ValueError as e:
        print("Error:", e)
