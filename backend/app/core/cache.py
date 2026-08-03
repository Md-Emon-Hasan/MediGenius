"""
MediGenius — core/cache.py
Simple in-memory TTL caching. Exact-match only — no semantic cache, see Phase 3 report
for why (near-identical embeddings, opposite correct answers, e.g. dosage vs pregnancy dosage).
"""

import re
import threading

from cachetools import TTLCache

from app.core import dosage_grounding

ANSWER_CACHE_TTL = 3600
ANSWER_CACHE_SIZE = 500
RETRIEVAL_CACHE_TTL = 21600
RETRIEVAL_CACHE_SIZE = 200

_answer_cache = TTLCache(maxsize=ANSWER_CACHE_SIZE, ttl=ANSWER_CACHE_TTL)
_retrieval_cache = TTLCache(maxsize=RETRIEVAL_CACHE_SIZE, ttl=RETRIEVAL_CACHE_TTL)
_lock = threading.Lock()


def _normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def is_cacheable(question: str) -> bool:
    """Dosage, pregnancy, paediatric, and drug-interaction questions never enter the cache."""
    return dosage_grounding.check_refusal(question) is None


def get_answer(question: str):
    if not is_cacheable(question):
        return None
    with _lock:
        return _answer_cache.get(_normalize_key(question))


def set_answer(question: str, value) -> None:
    if not is_cacheable(question):
        return
    with _lock:
        _answer_cache[_normalize_key(question)] = value


def get_retrieval(query: str):
    with _lock:
        return _retrieval_cache.get(_normalize_key(query))


def set_retrieval(query: str, value) -> None:
    with _lock:
        _retrieval_cache[_normalize_key(query)] = value
