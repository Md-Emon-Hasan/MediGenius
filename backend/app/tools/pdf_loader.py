from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages from PDF")
    return docs


def split_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", ". ", "\n", " "]
    )
    splits = text_splitter.split_documents(docs)
    print(f"Split into {len(splits)} chunks")
    return splits


def process_pdf(pdf_path):
    """Load and split PDF documents"""
    docs = load_pdf(pdf_path)
    doc_splits = split_documents(docs)
    return doc_splits
