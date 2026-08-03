"""
MediGenius — schemas/chat.py
Pydantic schemas for chat request and response.
"""

from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    source: str
    timestamp: str
    success: bool
    disclaimer: Optional[str] = None
    safety: Optional[dict] = None
    verification: Optional[dict] = None
