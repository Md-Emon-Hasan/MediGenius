"""
MediGenius — tools/memory_store.py
Semantic recall over past exchanges. Same Chroma persist directory and embedding
model as the medical-PDF vector store, in a separate collection so the two never mix.
Session-scoped only — recall never crosses between sessions.
"""

from typing import List

from app.core.config import VECTOR_STORE_DIR
from app.core.logging_config import logger
from app.tools.vector_store import get_embeddings

_memory_store = None


def get_memory_store():
    """Return a cached Chroma collection dedicated to conversation memory."""
    global _memory_store
    if _memory_store is None:
        from langchain_community.vectorstores import Chroma

        _memory_store = Chroma(
            collection_name="conversation_memory",
            persist_directory=VECTOR_STORE_DIR,
            embedding_function=get_embeddings(),
        )
        logger.info("Semantic memory store initialized")
    return _memory_store


def add_exchange(session_id: str, question: str, answer: str) -> None:
    """Store one Q&A pair for later recall. Never fails the request it's called from."""
    from langchain_core.documents import Document

    try:
        store = get_memory_store()
        store.add_documents([
            Document(
                page_content=f"Patient asked: {question}\nAnswer given: {answer}",
                metadata={"session_id": session_id},
            )
        ])
    except Exception as e:
        logger.error("memory_store: failed to store exchange: %s", str(e))


def recall(session_id: str, query: str, k: int) -> List[str]:
    """Return up to k past exchanges from this session relevant to the query."""
    try:
        store = get_memory_store()
        docs = store.similarity_search(query, k=k, filter={"session_id": session_id})
        return [d.page_content for d in docs]
    except Exception as e:
        logger.error("memory_store: recall failed: %s", str(e))
        return []


def delete_session_memory(session_id: str) -> None:
    """Purge every stored exchange for a session — called when a user deletes their history."""
    try:
        store = get_memory_store()
        store._collection.delete(where={"session_id": session_id})
    except Exception as e:
        logger.error("memory_store: delete failed: %s", str(e))
