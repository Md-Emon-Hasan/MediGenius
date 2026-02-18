"""
MediGenius — tools/llm_client.py
Groq LLM client singleton.
"""

from app.core.config import GROQ_API_KEY
from app.core.logging_config import logger

_llm_instance = None


def get_llm():
    """Return a cached ChatGroq LLM instance, or None if API key is missing."""
    global _llm_instance
    if _llm_instance is None:
        if not GROQ_API_KEY:
            logger.warning("GROQ_API_KEY not found in environment variables")
            return None
        from langchain_groq import ChatGroq

        _llm_instance = ChatGroq(
            api_key=GROQ_API_KEY,
            model_name="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=2048,
        )
        logger.info("LLM client initialized (Groq / llama-3.3-70b-versatile)")
    return _llm_instance
