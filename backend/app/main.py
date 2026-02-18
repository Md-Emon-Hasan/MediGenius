"""
MediGenius Backend — main.py
FastAPI application entry point: app setup, lifespan, and router registration.

Module layout:
  core/               — config, logging, state, workflow
  agents/             — 8 individual LangGraph agent nodes
  tools/              — LLM client, vector store, PDF loader, search tools
  db/                 — SQLAlchemy session factory
  models/             — ORM models
  schemas/            — Pydantic request/response schemas
  services/           — DatabaseService, ChatService
  api/v1/endpoints/   — health, chat, session route handlers
  api/v1/api.py       — router aggregator
  main.py             — FastAPI app + lifespan  ← you are here
"""

import os
import secrets
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

from app.api.v1.api import api_router
from app.core.config import CHAT_DB_PATH, PDF_PATH, VECTOR_STORE_DIR
from app.core.logging_config import logger
from app.services.chat_service import chat_service
from app.services.database_service import db_service
from app.tools.pdf_loader import process_pdf
from app.tools.vector_store import get_or_create_vectorstore


# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("Initializing MediGenius System...")

    db_service.init_db()
    logger.info("Database initialized at %s", CHAT_DB_PATH)

    if os.path.exists(PDF_PATH):
        logger.info("Processing PDF: %s", PDF_PATH)
        documents = process_pdf(PDF_PATH)
        get_or_create_vectorstore(documents)
        logger.info("Vector store ready at %s", VECTOR_STORE_DIR)
    else:
        logger.warning("PDF not found at %s — vector store skipped", PDF_PATH)

    chat_service.initialize_workflow()
    logger.info("MediGenius System Ready!")

    yield

    logger.info("Shutting down MediGenius...")


# ── FastAPI App ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="MediGenius API",
    description="AI-powered medical consultation system — Deep Modular + Agentic Architecture",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

# Register all API routes
app.include_router(api_router)


# ── Entry Point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
