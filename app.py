import os
from flask import Flask, request, jsonify, render_template
from langchain.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function
from query_data import query_rag
from populate_database import main

app = Flask(__name__)


CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Responde la pregunta utilizando solo el contexto de arriba: {question}

---

"""


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prompt = data.get('prompt')
    
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400
    
    # Generar respuesta usando RAG
    response = query_rag(prompt)
    
    return jsonify({"response": response})

if __name__ == '__main__':
    main()
    get_embedding_function()

    app.run(debug=True, port=5000)