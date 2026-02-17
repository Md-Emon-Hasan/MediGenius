"""Additional edge case tests for API routes"""
import os
import sys
from unittest.mock import patch

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))


def test_chat_with_header_session_id(test_client, mock_dependencies):
    """Test chat endpoint with X-Session-ID header"""
    with patch('services.chat_service.chat_service.process_message') as mock_process:
        mock_process.return_value = {
            "response": "Test response",
            "source": "Test",
            "timestamp": "10:00 AM",
            "success": True
        }
        response = test_client.post(
            "/api/chat",
            json={"message": "Hello"},
            headers={"X-Session-ID": "custom-session-id"}
        )
        assert response.status_code == 200


def test_get_history_with_header(test_client):
    """Test get history with X-Session-ID header"""
    with patch('services.database_service.db_service.get_chat_history') as mock_hist:
        mock_hist.return_value = []
        response = test_client.get("/api/history", headers={"X-Session-ID": "test-id"})
        assert response.status_code == 200


def test_delete_current_session(test_client):
    """Test deleting current session resets session ID"""
    with patch('services.database_service.db_service.delete_session'):
        # First create a session
        response = test_client.post("/api/new-chat")
        session_id = response.json()["session_id"]

        # Delete it
        response = test_client.delete(f"/api/session/{session_id}")
        assert response.status_code == 200
        assert response.json()["success"] is True


def test_clear_with_header(test_client):
    """Test clear endpoint with X-Session-ID header"""
    response = test_client.post("/api/clear", headers={"X-Session-ID": "test-id"})
    assert response.status_code == 200
    assert response.json()["message"] == "Conversation cleared"
