"""Tests for the audit log model and service method"""
import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import get_engine, get_session_factory  # noqa: E402
from app.services.database_service import DatabaseService  # noqa: E402

TEST_DB = "tests/test_database/test_audit.db"


@pytest.fixture(autouse=True)
def setup_teardown_db():
    os.makedirs(os.path.dirname(TEST_DB), exist_ok=True)
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass

    test_engine = get_engine(TEST_DB)
    test_session = get_session_factory(test_engine)
    service = DatabaseService(session_local=test_session, engine_instance=test_engine)
    service.init_db()

    yield service

    test_engine.dispose()
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass


def test_save_audit_log_minimal(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", source="Medical Literature Database", latency_ms=42.0)


def test_save_audit_log_full(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log(
        "sess-1",
        safety_category="crisis",
        refused_topic=None,
        figures_removed_count=2,
        source="Safety Router",
        model_used="groq/openai/gpt-oss-120b",
        latency_ms=5.5,
        degraded=False,
        cache_hit=False,
    )


def test_audit_log_does_not_store_message_text(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", source="AI Medical Knowledge")
    with service.get_session() as session:
        from app.models.audit_log import AuditLog
        row = session.query(AuditLog).first()
        assert not hasattr(row, "question")
        assert not hasattr(row, "answer")
        assert not hasattr(row, "content")
