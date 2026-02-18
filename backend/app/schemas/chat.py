"""
MediGenius — schemas/chat.py
Pydantic schemas for chat request and response.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    source: str
    timestamp: str
    success: bool
