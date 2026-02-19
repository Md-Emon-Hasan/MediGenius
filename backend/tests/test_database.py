"""Tests for database service — Deep Modular Architecture"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import get_engine, get_session_factory  # noqa: E402
from app.services.database_service import DatabaseService  # noqa: E402

TEST_DB = "tests/test_database/test_chat.db"


@pytest.fixture(autouse=True)
def setup_teardown_db():
    """Setup and teardown test database"""
    os.makedirs(os.path.dirname(TEST_DB), exist_ok=True)
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass

    test_engine = get_engine(TEST_DB)
    test_session = get_session_factory(test_engine)
    test_db_service = DatabaseService(session_local=test_session, engine_instance=test_engine)
    test_db_service.init_db()

    yield test_db_service

    test_engine.dispose()
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            print(f"Warning: Could not delete {TEST_DB} - file still in use")


def test_save_and_get_message(setup_teardown_db):
    db = setup_teardown_db
    session_id = "sess_1"

    db.save_message(session_id, "user", "Hello World")
    db.save_message(session_id, "assistant", "Hi there", "AI")

    history = db.get_chat_history(session_id)
    assert len(history) == 2
    assert history[0]["content"] == "Hello World"
    assert history[1]["source"] == "AI"


def test_get_all_sessions(setup_teardown_db):
    db = setup_teardown_db

    db.save_message("sess_1", "user", "msg1")
    db.save_message("sess_2", "user", "msg2")

    sessions = db.get_all_sessions()
    assert len(sessions) >= 2

    ids = [s["session_id"] for s in sessions]
    assert "sess_1" in ids
    assert "sess_2" in ids


def test_delete_session(setup_teardown_db):
    db = setup_teardown_db

    db.save_message("sess_to_del", "user", "delete me")
    assert len(db.get_chat_history("sess_to_del")) == 1

    db.delete_session("sess_to_del")
    assert len(db.get_chat_history("sess_to_del")) == 0
