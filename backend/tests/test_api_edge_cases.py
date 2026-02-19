"""Edge case tests for API routes — Deep Modular Architecture"""
from unittest.mock import patch

from app.services import chat_service, db_service


def test_chat_with_header_session_id(test_client, mock_dependencies):
    """Test chat endpoint with X-Session-ID header"""
    with patch.object(chat_service, 'process_message') as mock_process:
        mock_process.return_value = {
            "response": "Test response",
            "source": "Test",
            "timestamp": "10:00 AM",
            "success": True
        }
        response = test_client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
            headers={"X-Session-ID": "custom-session-id"}
        )
        assert response.status_code == 200


def test_get_history_with_header(test_client):
    """Test get history with X-Session-ID header"""
    with patch.object(db_service, 'get_chat_history') as mock_hist:
        mock_hist.return_value = []
        response = test_client.get("/api/v1/history", headers={"X-Session-ID": "test-id"})
        assert response.status_code == 200


def test_delete_current_session(test_client):
    """Test deleting current session resets session ID"""
    with patch.object(db_service, 'delete_session'):
        response = test_client.post("/api/v1/new-chat")
        session_id = response.json()["session_id"]

        response = test_client.delete(f"/api/v1/session/{session_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True


def test_clear_with_header(test_client):
    """Test clear endpoint with X-Session-ID header"""
    response = test_client.post("/api/v1/clear", headers={"X-Session-ID": "test-id"})
    assert response.status_code == 200
    assert response.json()["message"] == "Conversation cleared"
