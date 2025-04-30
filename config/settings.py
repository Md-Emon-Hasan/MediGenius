import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Document Processing
CHUNK_SIZE = 512
CHUNK_OVERLAP = 128

# Vector Store
PERSIST_DIR = BASE_DIR / "docs_db"
COLLECTION_NAME = "medical_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# LLM Settings
LLM_MODEL_NAME = "llama-3.3-70b-versatile"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# Search Settings
WIKI_TOP_K = 2
WIKI_MAX_CHARS = 2000