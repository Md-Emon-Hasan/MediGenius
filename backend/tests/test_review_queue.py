"""Tests for the audit_log migration and the clinician review queue"""
import os
import sys

import pytest
from sqlalchemy import text

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import get_engine, get_session_factory  # noqa: E402
from app.services.database_service import DatabaseService  # noqa: E402

TEST_DB = "tests/test_database/test_review.db"


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


def test_migration_adds_missing_columns_without_dropping_data(setup_teardown_db):
    service = setup_teardown_db
    with service.engine.begin() as conn:
        conn.execute(text("DROP TABLE audit_log"))
        conn.execute(text("CREATE TABLE audit_log (id INTEGER PRIMARY KEY, session_id VARCHAR(255), source VARCHAR(255))"))
        conn.execute(text("INSERT INTO audit_log (session_id, source) VALUES ('pre-existing-row', 'AI Medical Knowledge')"))

    service._migrate_audit_log_columns()

    with service.engine.connect() as conn:
        columns = {row[1] for row in conn.execute(text("PRAGMA table_info(audit_log)"))}
        rows = conn.execute(text("SELECT session_id, source FROM audit_log")).fetchall()

    for expected in ["model_fallback", "verification_risk", "needs_review", "review_status", "human_verdict", "reviewed_at"]:
        assert expected in columns
    assert rows == [("pre-existing-row", "AI Medical Knowledge")]


def test_needs_review_true_for_safety_category(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", safety_category="crisis")
    queue = service.get_review_queue()
    assert queue["total"] == 1


def test_needs_review_true_for_figures_removed(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", figures_removed_count=1)
    assert service.get_review_queue()["total"] == 1


def test_needs_review_true_for_model_fallback(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", model_fallback=True)
    assert service.get_review_queue()["total"] == 1


def test_needs_review_true_for_high_verification_risk(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", verification_risk="high")
    assert service.get_review_queue()["total"] == 1


def test_needs_review_false_for_ordinary_answer(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", source="AI Medical Knowledge", verification_risk="low")
    assert service.get_review_queue()["total"] == 0


def test_review_queue_pagination(setup_teardown_db):
    service = setup_teardown_db
    for i in range(5):
        service.save_audit_log(f"sess-{i}", safety_category="crisis")

    page1 = service.get_review_queue(page=1, page_size=2)
    assert page1["total"] == 5
    assert len(page1["items"]) == 2

    page3 = service.get_review_queue(page=3, page_size=2)
    assert len(page3["items"]) == 1


def test_review_queue_filters_by_status(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", safety_category="crisis")
    row_id = service.get_review_queue()["items"][0]["id"]
    service.submit_review(row_id, "looks fine", True)

    assert service.get_review_queue(status="pending")["total"] == 0
    assert service.get_review_queue(status="agreed")["total"] == 1


def test_submit_review_records_verdict_without_overwriting_original(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", safety_category="crisis", source="Safety Router")
    row_id = service.get_review_queue()["items"][0]["id"]

    updated = service.submit_review(row_id, "handled correctly", True)
    assert updated["human_verdict"] == "handled correctly"
    assert updated["review_status"] == "agreed"
    assert updated["reviewed_at"] is not None
    assert updated["safety_category"] == "crisis"
    assert updated["source"] == "Safety Router"


def test_submit_review_disagree(setup_teardown_db):
    service = setup_teardown_db
    service.save_audit_log("sess-1", safety_category="emergency")
    row_id = service.get_review_queue()["items"][0]["id"]

    updated = service.submit_review(row_id, "should have been flagged differently", False)
    assert updated["review_status"] == "disagreed"


def test_submit_review_missing_item_returns_none(setup_teardown_db):
    service = setup_teardown_db
    assert service.submit_review(9999, "verdict", True) is None


def test_stats_agreement_rate(setup_teardown_db):
    service = setup_teardown_db
    service.save_message("sess-1", "user", "hi")
    service.save_audit_log("sess-1", safety_category="crisis")
    service.save_audit_log("sess-1", safety_category="emergency")
    ids = [item["id"] for item in service.get_review_queue()["items"]]
    service.submit_review(ids[0], "agree", True)
    service.submit_review(ids[1], "disagree", False)

    stats = service.get_stats()
    assert stats["reviewed_count"] == 2
    assert stats["model_human_agreement_rate"] == 0.5
    assert stats["total_messages"] == 1


def test_stats_agreement_rate_none_when_nothing_reviewed(setup_teardown_db):
    service = setup_teardown_db
    stats = service.get_stats()
    assert stats["model_human_agreement_rate"] is None
