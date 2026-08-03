"""
MediGenius — api/v1/endpoints/review.py
Clinician review queue: GET /review, POST /review/{id}, GET /stats.
Unauthenticated — see README limitations, auth is required before this ships to real users.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.core.config import RATE_LIMIT
from app.core.rate_limit import limiter
from app.schemas.review import ReviewVerdictRequest
from app.services.database_service import db_service

router = APIRouter(tags=["Review"])


@router.get("/review")
async def get_review_queue(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    status: Optional[str] = Query(None),
):
    """Paginated queue of flagged answers awaiting clinician review."""
    return {**db_service.get_review_queue(page=page, page_size=page_size, status=status), "success": True}


@router.post("/review/{item_id}")
@limiter.limit(RATE_LIMIT)
async def submit_review(item_id: int, payload: ReviewVerdictRequest, request: Request):
    """Record a clinician's verdict alongside the flagged answer — never overwrites it."""
    updated = db_service.submit_review(item_id, payload.verdict, payload.reviewer_agrees)
    if not updated:
        raise HTTPException(status_code=404, detail="Review item not found")
    return {"item": updated, "success": True}


@router.get("/stats")
async def get_stats():
    """Aggregate counts including review backlog and model-vs-human agreement rate."""
    return {**db_service.get_stats(), "success": True}
