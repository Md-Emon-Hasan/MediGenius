import os

from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Global instances
_embeddings = None
_vectorstore = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embeddings


def get_or_create_vectorstore(documents=None, persist_dir='database/medical_db/'):
    """Get existing vectorstore or create new one if needed"""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    embeddings = get_embeddings()

    # Create directory if it doesn't exist
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    # Check if database files exist
    db_files_exist = False
    if os.path.exists(persist_dir):
        files = os.listdir(persist_dir)
        # Check for Chroma database files
        db_files_exist = any(f.endswith('.sqlite3') or f == 'chroma.sqlite3' or
                             f.startswith('index') for f in files)

    if db_files_exist:
        print("✓ Loading existing vector database...")
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"}
        )
        # Verify the database has content
        collection = _vectorstore._collection
        if collection.count() == 0:
            print("Vector database is empty, needs to be recreated")
            _vectorstore = None
            return None
        print(f"Loaded {collection.count()} documents from vector database")
    elif documents:
        print("Creating new vector database...")
        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"}
        )
        _vectorstore.persist()
        print(f"Created vector database with {len(documents)} documents")
    else:
        print("No existing database and no documents provided")
        return None

    return _vectorstore


def get_retriever(k=3):
    """Get retriever from existing vectorstore"""
    vectorstore = get_or_create_vectorstore()
    if vectorstore:
        return vectorstore.as_retriever(search_kwargs={'k': k})
    return None
