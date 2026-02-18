"""
MediGenius — tools/wikipedia_search.py
Wikipedia search wrapper singleton.
"""

from app.core.logging_config import logger

_wiki_wrapper = None


def get_wikipedia_wrapper():
    """Return a cached WikipediaAPIWrapper instance."""
    global _wiki_wrapper
    if _wiki_wrapper is None:
        from langchain_community.utilities.wikipedia import WikipediaAPIWrapper

        _wiki_wrapper = WikipediaAPIWrapper(
            top_k_results=2,
            doc_content_chars_max=2000,
            load_all_available_meta=True,
        )
        logger.info("Wikipedia search wrapper initialized")
    return _wiki_wrapper
