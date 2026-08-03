"""Tests for the review-queue and stats API endpoints"""
from unittest.mock import patch

from app.services.database_service import db_service


def test_get_review_queue_empty(test_client, mock_dependencies):
    with patch.object(db_service, 'get_review_queue', return_value={"items": [], "total": 0, "page": 1, "page_size": 20}):
        response = test_client.get("/api/v1/review")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["items"] == []


def test_get_review_queue_with_params(test_client, mock_dependencies):
    with patch.object(db_service, 'get_review_queue') as mock_queue:
        mock_queue.return_value = {"items": [], "total": 0, "page": 2, "page_size": 5}
        response = test_client.get("/api/v1/review?page=2&page_size=5&status=pending")
        assert response.status_code == 200
        mock_queue.assert_called_once_with(page=2, page_size=5, status="pending")


def test_submit_review_success(test_client, mock_dependencies):
    with patch.object(db_service, 'submit_review') as mock_submit:
        mock_submit.return_value = {"id": 1, "human_verdict": "looks right", "review_status": "agreed"}
        response = test_client.post("/api/v1/review/1", json={"verdict": "looks right", "reviewer_agrees": True})
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["item"]["review_status"] == "agreed"
        mock_submit.assert_called_once_with(1, "looks right", True)


def test_submit_review_not_found(test_client, mock_dependencies):
    with patch.object(db_service, 'submit_review', return_value=None):
        response = test_client.post("/api/v1/review/999", json={"verdict": "x", "reviewer_agrees": True})
        assert response.status_code == 404


def test_submit_review_validates_payload(test_client, mock_dependencies):
    response = test_client.post("/api/v1/review/1", json={"verdict": "", "reviewer_agrees": True})
    assert response.status_code == 422


def test_get_stats(test_client, mock_dependencies):
    with patch.object(db_service, 'get_stats') as mock_stats:
        mock_stats.return_value = {
            "total_messages": 10, "total_processed": 8, "pending_review": 2,
            "reviewed_count": 3, "model_human_agreement_rate": 0.67,
        }
        response = test_client.get("/api/v1/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["model_human_agreement_rate"] == 0.67
