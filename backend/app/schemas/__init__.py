"""Pydantic schemas for API validation"""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request schema for chat endpoint"""
    message: str


class ChatResponse(BaseModel):
    """Response schema for chat endpoint"""
    response: str
    source: str
    timestamp: str
    success: bool


class SessionResponse(BaseModel):
    """Response schema for session endpoints"""
    session_id: str
    preview: str
    last_active: str


class MessageResponse(BaseModel):
    """Response schema for message history"""
    role: str
    content: str
    source: str | None = None
    timestamp: str
