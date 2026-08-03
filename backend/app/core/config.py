"""
MediGenius — core/config.py
Environment variables and path constants.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
# backend/app/core/config.py -> backend/
_BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# Ensure logs and storage are inside backend directory
LOG_DIR = os.getenv("LOG_DIR", os.path.join(_BACKEND_DIR, "logs"))
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", os.path.join(_BACKEND_DIR, "storage", "chat_db", "medigenius.db"))
VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", os.path.join(_BACKEND_DIR, "storage", "vector_store"))
PDF_PATH = os.getenv("PDF_PATH", os.path.join(_BACKEND_DIR, "data", "medical_book.pdf"))

# ── API Keys ───────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── Rate Limiting ──────────────────────────────────────────────────────────────
# protects Groq/Tavily quota and avoids tripping DuckDuckGo's IP-block threshold
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "1") == "1"
RATE_LIMIT = os.getenv("RATE_LIMIT", "20/minute")

# ── Memory ─────────────────────────────────────────────────────────────────────
# caps how many recalled exchanges get injected into a prompt — unbounded recall quietly inflates every prompt
MAX_RECALLED_MEMORIES = int(os.getenv("MAX_RECALLED_MEMORIES", "3"))

# ── Model Gateway ──────────────────────────────────────────────────────────────
# llama-3.1-8b-instant and llama-3.3-70b-versatile are deprecated on Groq (shutdown 2026-08-16);
# these defaults are their vendor-recommended replacements — re-verify at console.groq.com/docs/models
SYNTHESIS_MODEL = os.getenv("SYNTHESIS_MODEL", "groq/openai/gpt-oss-120b")
REASONING_MODEL = os.getenv("REASONING_MODEL", "groq/openai/gpt-oss-120b")
CLASSIFICATION_MODEL = os.getenv("CLASSIFICATION_MODEL", "groq/openai/gpt-oss-20b")
