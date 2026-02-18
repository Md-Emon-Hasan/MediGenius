"""
MediGenius — tools/vector_store.py
ChromaDB vector store: embeddings, creation, loading, and retriever factory.
"""

import os
from typing import List, Optional

from langchain_core.documents import Document

from app.core.config import VECTOR_STORE_DIR
from app.core.logging_config import logger

_embeddings = None
_vectorstore = None


def get_embeddings():
    """Return a cached HuggingFace sentence-transformer embeddings instance."""
    global _embeddings
    if _embeddings is None:
        from langchain_huggingface.embeddings import HuggingFaceEmbeddings

        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        logger.info("Embeddings model loaded (all-MiniLM-L6-v2)")
    return _embeddings


def get_or_create_vectorstore(
    documents: Optional[List[Document]] = None,
    persist_dir: str = VECTOR_STORE_DIR,
):
    """Load existing ChromaDB vector store or create a new one from documents."""
    global _vectorstore

    if _vectorstore is not None:
        return _vectorstore

    from langchain_community.vectorstores import Chroma

    embeddings = get_embeddings()

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir)

    db_files_exist = any(
        f.endswith(".sqlite3") or f == "chroma.sqlite3" or f.startswith("index")
        for f in os.listdir(persist_dir)
    ) if os.path.exists(persist_dir) else False

    if db_files_exist:
        logger.info("Loading existing vector store from %s", persist_dir)
        _vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_metadata={"hnsw:space": "cosine"},
        )
        if _vectorstore._collection.count() == 0:
            logger.warning("Vector store is empty — needs to be recreated")
            _vectorstore = None
            return None
        logger.info(
            "Loaded %d documents from vector store", _vectorstore._collection.count()
        )
    elif documents:
        logger.info("Creating new vector store with %d documents", len(documents))
        _vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_metadata={"hnsw:space": "cosine"},
        )
        _vectorstore.persist()
    else:
        logger.warning("No existing vector store and no documents provided")
        return None

    return _vectorstore


def get_retriever(k: int = 3):
    """Return a retriever from the vector store, or None if unavailable."""
    vs = get_or_create_vectorstore()
    if vs:
        return vs.as_retriever(search_kwargs={"k": k})
    return None
