"""
MediGenius Backend - Main Application Entry Point
Industry-standard layered architecture with proper separation of concerns
"""
import os
import secrets
import sys
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware

# Add app directory to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
load_dotenv()

# Import services
from services.chat_service import chat_service  # noqa: E402
from services.database_service import db_service  # noqa: E402
from tools.pdf_loader import process_pdf  # noqa: E402
from tools.vector_store import get_or_create_vectorstore  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle events for the application"""
    # Startup
    print("Initializing MediGenius System...")

    # Initialize database
    db_service.init_db()
    print("Database initialized...")

    # Process PDF and create vector store
    pdf_path = os.getenv("PDF_PATH", "database/medical_book.pdf")
    if os.path.exists(pdf_path):
        print(f"Processing PDF: {pdf_path}")
        documents = process_pdf(pdf_path)
        get_or_create_vectorstore(documents)
        print("Vector store created...")
    else:
        print(f"Warning: PDF not found at {pdf_path}")

    # Initialize workflow
    chat_service.initialize_workflow()
    print("Workflow initialized...")

    print("MediGenius System Ready!")

    yield

    # Shutdown
    print("Shutting down MediGenius...")


# Create FastAPI app
app = FastAPI(
    title="MediGenius API",
    description="AI-powered medical consultation system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session Middleware
app.add_middleware(SessionMiddleware, secret_key=secrets.token_hex(32))

# Include API routes
from api import api_router  # noqa: E402

app.include_router(api_router, prefix="/api")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
