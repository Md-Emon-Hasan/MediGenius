"""
MediGenius — schemas/review.py
Pydantic schemas for the clinician review queue.
"""

from typing import Optional

from pydantic import BaseModel, Field


class ReviewVerdictRequest(BaseModel):
    verdict: str = Field(..., min_length=1, max_length=2000)
    reviewer_agrees: bool


class ReviewItemResponse(BaseModel):
    id: int
    session_id: str
    safety_category: Optional[str] = None
    refused_topic: Optional[str] = None
    figures_removed_count: int = 0
    source: Optional[str] = None
    model_used: Optional[str] = None
    model_fallback: bool = False
    verification_risk: Optional[str] = None
    degraded: bool = False
    needs_review: bool = False
    review_status: str = "pending"
    human_verdict: Optional[str] = None
    reviewed_at: Optional[str] = None
    timestamp: Optional[str] = None


class ReviewQueueResponse(BaseModel):
    items: list[ReviewItemResponse]
    total: int
    page: int
    page_size: int
