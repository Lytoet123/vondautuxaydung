import os
import logging
import hashlib
import json
from typing import Dict
from functools import lru_cache
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from flask import Flask, request, jsonify

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please check your .env file.")

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# Function to create or load FAISS index
def create_or_load_faiss_index(file_path, save_path):
    if os.path.exists(os.path.join(save_path, "index.faiss")):
        logging.info(f"Loading existing FAISS index from {save_path}")
        return FAISS.load_local(save_path, embedding_model)
    else:
        logging.info(f"Creating new FAISS index for {file_path}")
        with open(file_path, "rb") as file:
            reader = PdfReader(file)
            corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

        splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        chunks = splitter.split_text(corpus)

        vectors = FAISS.from_texts(chunks, embedding_model)
        vectors.save_local(save_path)
        return vectors

# Create FAISS index for each PDF file
file_paths = {
    "Vốn đầu tư": "data/vondautu.pdf",
    "Xây dựng": "data/xaydung.pdf"
}
faiss_indices = {}
for menu_item, file_path in file_paths.items():
    save_path = f"faiss_index/{menu_item}_index"
    faiss_indices[menu_item] = create_or_load_faiss_index(file_path, save_path)

# Flask app to provide an API
app = Flask(__name__)

@lru_cache(maxsize=1000)
def get_embedding(text):
    return embedding_model.embed_query(text)

def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode()).hexdigest()

def load_json(file_name: str) -> Dict:
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            json.dump({}, f)
    with open(file_name, "r") as f:
        return json.load(f)

def save_json(file_name: str, data: Dict):
    with open(file_name, "w") as f:
        json.dump(data, f)

def check_prompt_caching(query, cache_file):
    cache = load_json(cache_file)
    query_hash = hash_prompt(query)
    if query_hash in cache:
        logging.info(f"Cache hit for query: {query}")
        return cache[query_hash]
    logging.info(f"Cache miss for query: {query}")
    return None

def generate_answer_with_rag(query, vectors, previous_queries, max_tokens=150):
    llm = ChatOpenAI(model_name="gpt-4o", openai_api_key=api_key, max_tokens=max_tokens)
    contextual_query = f"{previous_queries} {query}"
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vectors.as_retriever(), return_source_documents=True)
    result = qa_chain({"query": contextual_query})
    return result['result']

@app.route('/answer', methods=['POST'])
def get_answer():
    data = request.get_json()
    query = data.get('query')
    selected_menu = data.get('selected_menu')
    chat_history = data.get('chat_history', [])

    if not query or not selected_menu:
        return jsonify({"error": "Missing 'query' or 'selected_menu' in request."}), 400

    vectors = faiss_indices.get(selected_menu)
    if not vectors:
        return jsonify({"error": f"No FAISS index found for menu '{selected_menu}'."}), 400

    if chat_history:
        previous_queries = ' '.join([f"User: {q}\nChatbot: {a}" for q, a in chat_history])
    else:
        previous_queries = ""

    # Check cache first
    cache_file = f"cache/{selected_menu.lower()}_cache.json"
    cached_answer = check_prompt_caching(query, cache_file)
    if cached_answer:
        logging.info(f"Returning cached answer for query: {query}")
        return jsonify({"answer": cached_answer})

    # Generate answer
    answer = generate_answer_with_rag(query, vectors, previous_queries, max_tokens=150)

    # Save to cache
    cache = load_json(cache_file)
    query_hash = hash_prompt(query)
    cache[query_hash] = answer
    save_json(cache_file, cache)

    response = {"answer": answer}
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
