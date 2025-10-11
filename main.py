from fastapi import FastAPI
from utils.model.rag import rag_answer
from utils.model.summarise import summarise

app = FastAPI()

@app.get("/")
def read_root():
    return "Welcome to the RAG and Summarization API!"

@app.get("/rag/")
def get_rag_answer(user_prompt: str):
    answer = rag_answer(user_prompt)
    return {"answer": answer}

@app.get("/summarise/")
def get_summary(text: str):
    summary = summarise(text)
    return {"summary": summary}