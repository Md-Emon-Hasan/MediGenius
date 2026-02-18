"""
MediGenius — api/v1/endpoints/session.py
Session management endpoints: /history, /sessions, /session/{id}.
"""

import uuid

from fastapi import APIRouter, Request

from app.services.database_service import db_service

router = APIRouter(tags=["Session"])


def _get_session_id(request: Request) -> str:
    """Get or create a session ID from X-Session-ID header or cookie session."""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


@router.get("/history")
async def get_history_endpoint(req: Request):
    """Return the chat history for the current session."""
    return {
        "messages": db_service.get_chat_history(_get_session_id(req)),
        "success": True,
    }


@router.get("/sessions")
async def get_sessions_endpoint():
    """Return a list of all chat sessions with previews."""
    return {"sessions": db_service.get_all_sessions(), "success": True}


@router.get("/session/{session_id}")
async def load_session_endpoint(session_id: str, req: Request):
    """Load a specific session by ID and set it as the active session."""
    req.session["session_id"] = session_id
    return {
        "messages": db_service.get_chat_history(session_id),
        "session_id": session_id,
        "success": True,
    }


@router.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str, req: Request):
    """Delete a session and reset the active session if it matches."""
    db_service.delete_session(session_id)
    if req.session.get("session_id") == session_id:
        req.session["session_id"] = str(uuid.uuid4())
    return {"message": "Session deleted", "success": True}
