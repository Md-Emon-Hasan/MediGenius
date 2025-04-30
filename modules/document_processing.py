import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger("medical_chatbot")

class DocumentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", ". ", "\n", " "]
        )

    def load_and_split_documents(self, file_path: str):
        try:
            logger.info(f"Loading documents from {file_path}")
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            
            logger.info(f"Splitting documents into chunks")
            splits = self.text_splitter.split_documents(docs)
            logger.info(f"Created {len(splits)} document chunks")
            
            return splits
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            raise