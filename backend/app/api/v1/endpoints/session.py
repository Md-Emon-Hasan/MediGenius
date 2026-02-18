"""Session endpoints for v1 API"""
from fastapi import APIRouter, Request
from services.database_service import db_service
import uuid

router = APIRouter()

def get_session_id(request: Request) -> str:
    """Get or create session ID"""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id

    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]

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
