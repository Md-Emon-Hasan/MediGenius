"""API routes for MediGenius"""
import uuid

from fastapi import APIRouter, HTTPException, Request
from schemas import ChatRequest, ChatResponse
from services.chat_service import chat_service
from services.database_service import db_service

router = APIRouter(prefix="/api")


def get_session_id(request: Request) -> str:
    """Get or create session ID"""
    # Try to get from header first (for API clients)
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id

    # Fallback to cookie session
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "MediGenius Backend"}


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Chat endpoint for processing messages"""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")

    session_id = get_session_id(req)
    result = await chat_service.process_message(session_id, request.message)
    return result


@router.get("/history")
async def get_history_endpoint(req: Request):
    """Get chat history for current session"""
    session_id = get_session_id(req)
    messages = db_service.get_chat_history(session_id)
    return {"messages": messages, "success": True}


@router.get("/sessions")
async def get_sessions_endpoint():
    """Get all chat sessions"""
    sessions = db_service.get_all_sessions()
    return {"sessions": sessions, "success": True}


@router.get("/session/{session_id}")
async def load_session_endpoint(session_id: str, req: Request):
    """Load a specific session"""
    req.session["session_id"] = session_id
    messages = db_service.get_chat_history(session_id)
    return {
        "messages": messages,
        "session_id": session_id,
        "success": True
    }


@router.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str, req: Request):
    """Delete a session"""
    db_service.delete_session(session_id)

    # If current session deleted, reset
    current = req.session.get("session_id")
    if current == session_id:
        req.session["session_id"] = str(uuid.uuid4())

    return {"message": "Session deleted", "success": True}


@router.post("/clear")
async def clear_endpoint(req: Request):
    """Clear current conversation"""
    session_id = get_session_id(req)
    chat_service.clear_conversation(session_id)
    return {"message": "Conversation cleared", "success": True}


@router.post("/new-chat")
async def new_chat_endpoint(req: Request):
    """Create a new chat session"""
    new_id = str(uuid.uuid4())
    req.session["session_id"] = new_id
    return {
        "message": "New chat created",
        "session_id": new_id,
        "success": True
    }
