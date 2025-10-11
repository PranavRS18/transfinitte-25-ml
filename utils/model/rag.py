import utils.model.init as init

def rag_answer(user_prompt):
    query, retrieved_documents, system_prompt, context = init.retrieve("What is Artificial Intelligence?")
    data = {
        "model": "llama3.2:1b",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": 200
    }

    response = requests.post("http://localhost:11434/v1/chat/completions",
                            headers={"Content-Type": "application/json"},
                            data=json.dumps(data))

    return response.json()["choices"][0]["message"]["content"]