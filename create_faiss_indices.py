import os
import logging
from PyPDF2 import PdfReader, PdfReadError  # Import PdfReadError
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv
#from tenacity import retry, stop_after_attempt, wait_exponential # Nếu muốn dùng retry

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') # Dùng nhất quán OPENAI_API_KEY
if not OPENAI_API_KEY:
    raise ValueError("API key not found. Please check your .env file.")

# Initialize the embedding model
try:
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)
except Exception as e:
    logging.error(f"Error initializing OpenAIEmbeddings: {e}", exc_info=True)
    raise # Re-raise nếu không thể khởi tạo embedding model


# Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Function to create FAISS index
#@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)) # Nếu dùng tenacity
def create_faiss_index(file_path: str, save_path: str):
    logging.info(f"Creating FAISS index for {file_path}")
    try:
        with open(file_path, "rb") as file:
            try:
                reader = PdfReader(file)
                corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])
            except PdfReadError as e: # Bắt lỗi PyPDF2 cụ thể
                logging.error(f"Error reading PDF {file_path}: {e}")
                raise

        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_text(corpus)
        
        try:
            vectors = FAISS.from_texts(chunks, embedding_model)
            vectors.save_local(save_path)
            logging.info(f"FAISS index saved to {save_path}")
        except Exception as e:
            logging.error(f"Error creating/saving FAISS index for {file_path}: {e}", exc_info=True) # log chi tiết hơn
            raise


    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise # Re-raise để dừng chương trình, hoặc xử lý khác
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise


# Define file paths
file_paths = {
    "Vốn đầu tư": "data/vondautu.pdf",
    "Xây dựng": "data/xaydung.pdf"
}

# Create faiss_index directory if it doesn't exist
os.makedirs("faiss_index", exist_ok=True)

# Create FAISS indices
for menu_item, file_path in file_paths.items():
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        continue # Bỏ qua file này, tiếp tục với file khác

    index_path = os.path.join("faiss_index", f"{menu_item.lower()}_index") # Dùng os.path.join
    try:
        create_faiss_index(file_path, index_path)
    except Exception:
        # Đã log lỗi bên trong create_faiss_index, có thể không cần làm gì thêm ở đây
        pass
