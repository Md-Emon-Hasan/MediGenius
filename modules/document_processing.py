import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

# Initialize logger for the medical chatbot system
logger = logging.getLogger("medical_chatbot")

class DocumentProcessor:
    """
    Handles loading and preprocessing of PDF documents by splitting them into 
    smaller, overlapping text chunks suitable for LLM input.
    """

    def __init__(self):
        # Initialize a recursive character-based text splitter with custom chunk size, overlap, and separators
        self.text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", ". ", "\n", " "]  # Priority-based text split strategy
        )

    def load_and_split_documents(self, file_path: str):
        """
        Loads a PDF file and splits it into manageable chunks for downstream LLM processing.

        Args:
            file_path (str): Path to the PDF file.

        Returns:
            List[Document]: A list of split document chunks.
        """
        try:
            logger.info(f"Loading documents from {file_path}")
            loader = PyPDFLoader(file_path)  # Load PDF using Langchain-compatible loader
            docs = loader.load()  # Extract all document pages
            
            logger.info(f"Splitting documents into chunks")
            splits = self.text_splitter.split_documents(docs)  # Perform chunking based on the tokenizer and strategy
            logger.info(f"Created {len(splits)} document chunks")  # Log the number of generated chunks
            
            return splits
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")  # Log any error encountered during loading or splitting
            raise  # Re-raise the exception to surface issues upstream
