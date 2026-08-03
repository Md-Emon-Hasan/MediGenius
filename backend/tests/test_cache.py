"""Tests for the exact-match answer/retrieval cache"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core import cache  # noqa: E402


def test_answer_cache_roundtrip():
    cache.set_answer("what is diabetes", "diabetes info")
    assert cache.get_answer("what is diabetes") == "diabetes info"


def test_answer_cache_key_normalized():
    cache.set_answer("  What IS Diabetes?  ".strip(), "diabetes info")
    assert cache.get_answer("what is diabetes?") == cache.get_answer("  What IS Diabetes?  ".strip())


def test_answer_cache_miss_returns_none():
    assert cache.get_answer("never asked before") is None


def test_dangerous_pairs_do_not_collide():
    # same-ish embedding neighborhood, opposite correct answers — must never share a cache entry
    pairs = [
        ("is ibuprofen safe?", "is ibuprofen safe during pregnancy?"),
        ("paracetamol dose", "paracetamol dose for a 2 year old"),
        ("can i take ibuprofen with aspirin", "can i take ibuprofen or aspirin"),
    ]
    for plain, sensitive in pairs:
        assert cache._normalize_key(plain) != cache._normalize_key(sensitive)


def test_dosage_sensitive_question_never_cached():
    cache.set_answer("paracetamol dose for a 2 year old", "should never be stored")
    assert cache.get_answer("paracetamol dose for a 2 year old") is None


def test_pregnancy_question_never_cached():
    cache.set_answer("is ibuprofen safe during pregnancy?", "should never be stored")
    assert cache.get_answer("is ibuprofen safe during pregnancy?") is None


def test_plain_question_is_cacheable():
    assert cache.is_cacheable("what is diabetes") is True
    assert cache.is_cacheable("is ibuprofen safe during pregnancy?") is False


def test_retrieval_cache_roundtrip():
    cache.set_retrieval("fever symptoms", ["doc1", "doc2"])
    assert cache.get_retrieval("fever symptoms") == ["doc1", "doc2"]
