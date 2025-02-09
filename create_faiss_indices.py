import os
import logging
from PyPDF2 import PdfReader
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load environment variables
load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
if not api_key:
    raise ValueError("API key not found. Please check your .env file.")

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=api_key)

# Function to create FAISS index
def create_faiss_index(file_path: str, save_path: str):
    logging.info(f"Creating FAISS index for {file_path}")
    with open(file_path, "rb") as file:
        reader = PdfReader(file)
        corpus = ''.join([p.extract_text() for p in reader.pages if p.extract_text()])

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(corpus)

    vectors = FAISS.from_texts(chunks, embedding_model)
    vectors.save_local(save_path)
    logging.info(f"FAISS index saved to {save_path}")

# Define file paths
file_paths = {
    "Vốn đầu tư": "data/vondautu.pdf",
    "Xây dựng": "data/xaydung.pdf"
}

# Create faiss_index directory if it doesn't exist
os.makedirs("faiss_index", exist_ok=True)

# Create FAISS indices
for menu_item, file_path in file_paths.items():
    index_path = f"faiss_index/{menu_item.lower()}_index"
    create_faiss_index(file_path, index_path)
