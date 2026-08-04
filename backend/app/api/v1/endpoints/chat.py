"""
MediGenius — api/v1/endpoints/chat.py
Chat-related endpoints: /chat, /chat/stream, /clear, /new-chat.
"""

import json
import uuid

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import RATE_LIMIT
from app.core.logging_config import logger
from app.core.rate_limit import limiter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.chat_service import chat_service

router = APIRouter(tags=["Chat"])


def _get_session_id(request: Request) -> str:
    """Get or create a session ID from X-Session-ID header or cookie session."""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id
    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]


@router.post("/chat", response_model=ChatResponse)
@limiter.limit(RATE_LIMIT)
async def chat_endpoint(payload: ChatRequest, request: Request):
    """Process a user message through the agentic pipeline."""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    session_id = _get_session_id(request)
    return await chat_service.process_message(session_id, payload.message)


@router.post("/chat/stream")
@limiter.limit(RATE_LIMIT)
async def chat_stream_endpoint(payload: ChatRequest, request: Request):
    """Same as /chat, but streams a Server-Sent Event per pipeline stage before the final answer."""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")
    session_id = _get_session_id(request)

    async def event_source():
        try:
            async for event in chat_service.process_message_stream(session_id, payload.message):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception:
            logger.error("chat_stream: pipeline failed mid-stream", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'message': 'Something went wrong. Please try again.'})}\n\n"

    return StreamingResponse(
        event_source(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/clear")
async def clear_endpoint(req: Request):
    """Clear the in-memory conversation state for the current session."""
    chat_service.clear_conversation(_get_session_id(req))
    return {"message": "Conversation cleared", "success": True}


@router.post("/new-chat")
async def new_chat_endpoint(req: Request):
    """Create a new chat session with a fresh session ID."""
    new_id = str(uuid.uuid4())
    req.session["session_id"] = new_id
    return {"message": "New chat created", "session_id": new_id, "success": True}
