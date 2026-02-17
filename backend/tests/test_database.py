"""Tests for database service"""
import os
import sys

import pytest

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from services.database_service import DatabaseService  # noqa: E402

# Use test DB
TEST_DB = "test_chat.db"


@pytest.fixture(autouse=True)
def setup_teardown_db():
    """Setup and teardown test database"""
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)

    # Create test database service
    test_db_service = DatabaseService(db_path=TEST_DB)
    test_db_service.init_db()

    yield test_db_service

    # Teardown
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_save_and_get_message(setup_teardown_db):
    """Test saving and retrieving messages"""
    db = setup_teardown_db
    session_id = "sess_1"

    db.save_message(session_id, "user", "Hello World")
    db.save_message(session_id, "assistant", "Hi there", "AI")

    history = db.get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "Hello World"
    assert history[1]["source"] == "AI"


def test_get_all_sessions(setup_teardown_db):
    """Test retrieving all sessions"""
    db = setup_teardown_db

    db.save_message("sess_1", "user", "msg1")
    db.save_message("sess_2", "user", "msg2")

    sessions = db.get_all_sessions()
    assert len(sessions) >= 2

    ids = [s["session_id"] for s in sessions]
    assert "sess_1" in ids
    assert "sess_2" in ids


def test_delete_session(setup_teardown_db):
    """Test deleting a session"""
    db = setup_teardown_db

    db.save_message("sess_to_del", "user", "delete me")
    assert len(db.get_chat_history("sess_to_del")) == 1

    db.delete_session("sess_to_del")
    assert len(db.get_chat_history("sess_to_del")) == 0
