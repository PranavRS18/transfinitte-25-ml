import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import requests, json

def embedding(documents):
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    document_embeddings = embedding_model.encode(documents, convert_to_numpy = True)

    dimension = document_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(document_embeddings))

def retrieve(query, top_k = 2):
    query_vector = embedding_model.encode([query], convert_to_numpy = True)

    D, I = index.search(np.array(query_vector), top_k)
    retrieved_documents = [documents[i] for i in I[0]]

    system_prompt = "You are a helpful assistant. Use the following context to answer the question.\n\n"
    context = "\n\n".join(retrieved_documents)

    return query, retrieved_documents, system_prompt, context

