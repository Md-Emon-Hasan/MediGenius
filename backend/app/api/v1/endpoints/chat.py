"""Chat endpoints for v1 API"""
import uuid
from fastapi import APIRouter, HTTPException, Request
from schemas import ChatRequest, ChatResponse
from services.chat_service import chat_service
from services.database_service import db_service

router = APIRouter()

def get_session_id(request: Request) -> str:
    """Get or create session ID"""
    session_id = request.headers.get("X-Session-ID")
    if session_id:
        return session_id

    if "session_id" not in request.session:
        request.session["session_id"] = str(uuid.uuid4())
    return request.session["session_id"]

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, req: Request):
    """Chat endpoint for processing messages"""
    if not chat_service.workflow_app:
        raise HTTPException(status_code=503, detail="System not initialized")

    session_id = get_session_id(req)
    result = await chat_service.process_message(session_id, request.message)
    return result

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
