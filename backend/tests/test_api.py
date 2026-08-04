import json
from unittest.mock import patch

from app.services import chat_service, db_service


def _parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        for line in block.splitlines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


def test_health_check(test_client):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "MediGenius Backend v2"}


def test_new_chat(test_client):
    response = test_client.post("/api/v1/new-chat")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "session_id" in data
    assert data["message"] == "New chat created"


def test_get_sessions(test_client):
    with patch.object(db_service, 'get_all_sessions') as mock_get:
        mock_get.return_value = [{"session_id": "123", "preview": "hi", "last_active": "2024-01-01"}]
        response = test_client.get("/api/v1/sessions")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert len(data["sessions"]) == 1
        assert data["sessions"][0]["session_id"] == "123"


def test_chat_flow_success(test_client, mock_dependencies):
    mock_dependencies["workflow_app"].ainvoke.return_value = {
        "generation": "Test connection successful",
        "source": "Mock Brain",
        "llm_success": True
    }

    response = test_client.post("/api/v1/new-chat")
    session_id = response.json()["session_id"]

    with patch("app.services.chat_service.memory_store.add_exchange"):
        chat_response = test_client.post(
            "/api/v1/chat",
            json={"message": "Hello AI"},
            headers={"X-Session-ID": session_id}
        )

    assert chat_response.status_code == 200
    data = chat_response.json()
    assert data["success"] is True
    assert data["response"] == "Test connection successful"
    assert data["source"] == "Mock Brain"


def test_chat_stream_flow_success(test_client, mock_dependencies):
    response = test_client.post("/api/v1/new-chat")
    session_id = response.json()["session_id"]

    with patch("app.services.chat_service.memory_store.add_exchange"):
        stream_response = test_client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello AI"},
            headers={"X-Session-ID": session_id},
        )

    assert stream_response.status_code == 200
    events = _parse_sse(stream_response.text)
    assert len(events) >= 2
    assert events[0]["type"] == "stage"
    final_events = [e for e in events if e["type"] == "final"]
    assert len(final_events) == 1
    assert final_events[0]["payload"]["response"] == "Test response from AI"


def test_chat_stream_mid_stream_failure_yields_error_event(test_client, mock_dependencies):
    async def failing_stream(session_id, message):
        yield {"type": "stage", "stage": "supervisor", "label": "Understanding your question"}
        raise RuntimeError("boom")

    with patch.object(chat_service, 'process_message_stream', failing_stream):
        response = test_client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello AI"},
        )

    assert response.status_code == 200
    events = _parse_sse(response.text)
    assert events[0]["type"] == "stage"
    assert events[-1]["type"] == "error"


def test_chat_stream_system_not_initialized(test_client):
    with patch.object(chat_service, 'workflow_app', None):
        response = test_client.post(
            "/api/v1/chat/stream",
            json={"message": "Hello"},
        )
        assert response.status_code == 503


def test_chat_flow_system_not_initialized(test_client):
    with patch.object(chat_service, 'workflow_app', None):
        response = test_client.post(
            "/api/v1/chat",
            json={"message": "Hello"},
        )
        assert response.status_code == 503
        assert response.json()["detail"] == "System not initialized"


def test_get_history(test_client):
    with patch.object(db_service, 'get_chat_history') as mock_hist:
        mock_hist.return_value = [{"role": "user", "content": "hi"}]
        response = test_client.get("/api/v1/history", headers={"X-Session-ID": "test-sess"})
        assert response.status_code == 200
        assert len(response.json()["messages"]) == 1


def test_load_session(test_client):
    with patch.object(db_service, 'get_chat_history') as mock_hist:
        mock_hist.return_value = []
        response = test_client.get("/api/v1/session/test-session-id")
        assert response.status_code == 200
        assert response.json()["session_id"] == "test-session-id"


def test_delete_session(test_client):
    with patch.object(db_service, 'delete_session') as mock_del:
        response = test_client.delete("/api/v1/session/test-id")
        assert response.status_code == 200
        assert response.json()["message"] == "Session deleted"
        mock_del.assert_called_once_with("test-id")


def test_clear_conversation(test_client):
    response = test_client.post("/api/v1/clear", headers={"X-Session-ID": "test-id"})
    assert response.status_code == 200
    assert response.json()["message"] == "Conversation cleared"
