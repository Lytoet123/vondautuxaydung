import os
import logging
import traceback
from typing import Dict, List, Tuple
from functools import lru_cache
from dotenv import load_dotenv
from PyPDF2 import PdfReader, PdfReadError
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
embedding_model = None
llm = None
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
    logger.error(f"Error initializing models: {str(e)}", exc_info=True) # Thêm exc_info=True
    # Có thể trả về một thông báo lỗi cho người dùng nếu cần
    # raise # Hoặc re-raise nếu không thể xử lý

class DocumentProcessor:
    def __init__(self, embedding_model: OpenAIEmbeddings):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200
        )
        self.embedding_model = embedding_model # Dependency injection
        
    def process_pdf(self, file_path: str) -> FAISS:
        """Process PDF file and create FAISS index"""
        index_path = os.path.join("faiss_index", f"{os.path.basename(file_path)}_index") # Sử dụng os.path.join
        try:
            # Check if FAISS index already exists
            if os.path.exists(os.path.join(index_path, "index.faiss")):
                logger.info(f"Loading existing index from {index_path}")
                return FAISS.load_local(index_path, self.embedding_model)
            
            # Create new index
            logger.info(f"Creating new index for {file_path}")
            with open(file_path, "rb") as file:
                try:
                    pdf = PdfReader(file)
                    text = ' '.join(page.extract_text() for page in pdf.pages)
                except PdfReadError as e:
                    logger.error(f"Error reading PDF {file_path}: {str(e)}", exc_info=True)
                    raise
                
            chunks = self.text_splitter.split_text(text)
            vectorstore = FAISS.from_texts(chunks, self.embedding_model)
            
            # Save index
            os.makedirs(index_path, exist_ok=True)
            vectorstore.save_local(index_path)
            
            return vectorstore
        except FileNotFoundError as e:
            logger.error(f"File not found: {file_path}: {str(e)}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}", exc_info=True)
            raise

class QASystem:
    def __init__(self, vectorstore: FAISS, llm: ChatOpenAI):
        self.vectorstore = vectorstore
        self.llm = llm # Dependency injection
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
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
            return response['result'], response['source_documents'] # Trả về cả source documents
        except Exception as e:
            logger.error(f"Error getting answer: {str(e)}", exc_info=True)
            return "Xin lỗi, tôi không thể trả lời câu hỏi này lúc này.", []

# Initialize document processor and load documents
processor = None # Khởi tạo bên ngoài khối try
qa_systems = {}
try:
    if embedding_model is None or llm is None:
        raise ValueError("Embedding model or LLM was not initialized properly.")

    processor = DocumentProcessor(embedding_model)

    # Define document paths
    DOCUMENTS = {
        "Vốn đầu tư": "data/vondautu.pdf",
        "Xây dựng": "data/xaydung.pdf"
    }

    # Load all documents
    for topic, path in DOCUMENTS.items():
        try:
            vectorstore = processor.process_pdf(path)
            qa_systems[topic] = QASystem(vectorstore, llm)
        except Exception as e:
            logger.error(f"Error loading document {topic}: {str(e)}", exc_info=True)

except Exception as e:
    logger.critical(f"Failed to initialize application: {str(e)}", exc_info=True)
    # Xử lý lỗi khởi tạo nghiêm trọng (ví dụ: dừng ứng dụng)
    raise

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
        answer, source_documents = qa_system.get_answer(query, context) # Lấy cả source documents
        
        return jsonify({"answer": answer, "sources": source_documents}) # Trả về cả answer và sources
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
