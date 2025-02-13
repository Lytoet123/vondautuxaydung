import os
import logging
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
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

# Initialize models and embeddings
try:
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENAI_API_KEY
    )
    
    llm = ChatOpenAI(
        model_name="gpt-4",
        openai_api_key=OPENAI_API_KEY,
        temperature=0.7
    )
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        
    def process_pdf(self, file_path: str) -> FAISS:
        """Process PDF file and create FAISS index"""
        try:
            # Check if FAISS index already exists
            index_path = f"faiss_index/{os.path.basename(file_path)}_index"
            if os.path.exists(os.path.join(index_path, "index.faiss")):
                logger.info(f"Loading existing index from {index_path}")
                return FAISS.load_local(index_path, embedding_model)
            
            # Create new index
            logger.info(f"Creating new index for {file_path}")
            with open(file_path, "rb") as file:
                pdf = PdfReader(file)
                text = ' '.join(page.extract_text() for page in pdf.pages)
                
            chunks = self.text_splitter.split_text(text)
            vectorstore = FAISS.from_texts(chunks, embedding_model)
            
            # Save index
            os.makedirs(index_path, exist_ok=True)
            vectorstore.save_local(index_path)
            
            return vectorstore
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise

class QASystem:
    def __init__(self, vectorstore: FAISS):
        self.vectorstore = vectorstore
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_kwargs={"k": 3}
            ),
            return_source_documents=True
        )
    
    def get_answer(self, query: str, context: str = "") -> str:
        """Get answer for query with optional context"""
        try:
            full_query = f"{context} {query}".strip()
            response = self.qa_chain({"query": full_query})
            return response['result']
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}")
            return "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này."

# Initialize document processor and load documents
processor = DocumentProcessor()
qa_systems = {}

# Define document paths
DOCUMENTS = {
    "Vốn đầu tư": "data/vondautu.pdf",
    "Xây dựng": "data/xaydung.pdf"
}

# Load all documents
for topic, path in DOCUMENTS.items():
    try:
        vectorstore = processor.process_pdf(path)
        qa_systems[topic] = QASystem(vectorstore)
    except Exception as e:
        logger.error(f"Error loading document {topic}: {str(e)}")

@app.route('/answer', methods=['POST'])
def answer():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        query = data.get('query')
        selected_menu = data.get('selected_menu')
        chat_history = data.get('chat_history', [])
        
        if not query or not selected_menu:
            return jsonify({"error": "Missing required fields"}), 400
            
        if selected_menu not in qa_systems:
            return jsonify({"error": "Invalid topic selected"}), 400
            
        # Build context from chat history
        context = " ".join(
            f"Q: {q} A: {a}" for q, a in chat_history[-3:]
        ) if chat_history else ""
        
        # Get answer
        qa_system = qa_systems[selected_menu]
        answer = qa_system.get_answer(query, context)
        
        return jsonify({"answer": answer})
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
