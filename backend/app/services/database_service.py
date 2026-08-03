"""
MediGenius — services/database_service.py
DatabaseService: all CRUD operations for chat history.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import delete, desc, func, inspect, select, text
from sqlalchemy.orm import Session

from app.core.logging_config import logger
from app.db.session import SessionLocal, engine
from app.models.audit_log import AuditLog
from app.models.message import Base, Message
from app.tools import memory_store

# added after audit_log already shipped — ALTER TABLE keeps existing rows instead of a destructive recreate
_AUDIT_LOG_NEW_COLUMNS = {
    "model_fallback": "BOOLEAN DEFAULT 0",
    "verification_risk": "VARCHAR(20)",
    "needs_review": "BOOLEAN DEFAULT 0",
    "review_status": "VARCHAR(20) DEFAULT 'pending'",
    "human_verdict": "TEXT",
    "reviewed_at": "DATETIME",
}


class DatabaseService:
    """All database CRUD operations for chat history."""

    def __init__(self, session_local=None, engine_instance=None):
        self.SessionLocal = session_local or SessionLocal
        self.engine = engine_instance or engine
        logger.info("DatabaseService initialized")

    def init_db(self) -> None:
        """Create all tables if they don't exist, then migrate audit_log for pre-existing rows."""
        logger.info("Initializing database tables...")
        Base.metadata.create_all(bind=self.engine)
        self._migrate_audit_log_columns()

    def _migrate_audit_log_columns(self) -> None:
        inspector = inspect(self.engine)
        if "audit_log" not in inspector.get_table_names():
            return
        existing = {col["name"] for col in inspector.get_columns("audit_log")}
        with self.engine.begin() as conn:
            for name, ddl in _AUDIT_LOG_NEW_COLUMNS.items():
                if name not in existing:
                    logger.info("Migrating audit_log: adding column %s", name)
                    conn.execute(text(f"ALTER TABLE audit_log ADD COLUMN {name} {ddl}"))

    def get_session(self) -> Session:
        return self.SessionLocal()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        source: Optional[str] = None,
    ) -> None:
        logger.debug("Saving %s message for session %s...", role, session_id[:8])
        with self.get_session() as session:
            session.add(
                Message(
                    session_id=session_id, role=role, content=content, source=source
                )
            )
            session.commit()

    def get_chat_history(self, session_id: str) -> List[Dict]:
        with self.get_session() as session:
            stmt = (
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.timestamp)
            )
            return [msg.to_dict() for msg in session.execute(stmt).scalars().all()]

    def get_all_sessions(self) -> List[Dict]:
        with self.get_session() as session:
            latest_sub = (
                select(
                    Message.session_id,
                    func.max(Message.timestamp).label("max_ts"),
                )
                .where(Message.role == "user")
                .group_by(Message.session_id)
                .subquery()
            )
            stmt = (
                select(Message.session_id, Message.content, Message.timestamp)
                .join(
                    latest_sub,
                    (Message.session_id == latest_sub.c.session_id)
                    & (Message.timestamp == latest_sub.c.max_ts),
                )
                .order_by(desc(Message.timestamp))
            )
            return [
                {
                    "session_id": row[0],
                    "preview": row[1][:50] + "..." if len(row[1]) > 50 else row[1],
                    "last_active": row[2].isoformat() if row[2] else None,
                }
                for row in session.execute(stmt).all()
            ]

    def delete_session(self, session_id: str) -> None:
        logger.info("Deleting session %s...", session_id[:8])
        with self.get_session() as session:
            session.execute(delete(Message).where(Message.session_id == session_id))
            session.commit()
        memory_store.delete_session_memory(session_id)

    def save_audit_log(self, session_id: str, **fields) -> None:
        fields.setdefault("needs_review", self._needs_review(fields))
        with self.get_session() as session:
            session.add(AuditLog(session_id=session_id, **fields))
            session.commit()

    @staticmethod
    def _needs_review(fields: Dict) -> bool:
        return bool(
            fields.get("safety_category")
            or fields.get("refused_topic")
            or (fields.get("figures_removed_count") or 0) > 0
            or fields.get("model_fallback")
            or fields.get("verification_risk") == "high"
        )

    def get_review_queue(self, page: int = 1, page_size: int = 20, status: Optional[str] = None) -> Dict:
        with self.get_session() as session:
            stmt = select(AuditLog).where(AuditLog.needs_review.is_(True))
            if status:
                stmt = stmt.where(AuditLog.review_status == status)
            total = len(session.execute(stmt).scalars().all())
            stmt = stmt.order_by(desc(AuditLog.timestamp)).offset((page - 1) * page_size).limit(page_size)
            items = [row.to_dict() for row in session.execute(stmt).scalars().all()]
            return {"items": items, "total": total, "page": page, "page_size": page_size}

    def submit_review(self, item_id: int, verdict: str, reviewer_agrees: bool) -> Optional[Dict]:
        with self.get_session() as session:
            row = session.get(AuditLog, item_id)
            if not row:
                return None
            row.human_verdict = verdict
            row.review_status = "agreed" if reviewer_agrees else "disagreed"
            row.reviewed_at = datetime.utcnow()
            session.commit()
            return row.to_dict()

    def get_stats(self) -> Dict:
        with self.get_session() as session:
            total_messages = session.execute(select(func.count()).select_from(Message)).scalar()
            total_processed = session.execute(select(func.count()).select_from(AuditLog)).scalar()
            pending_review = session.execute(
                select(func.count()).select_from(AuditLog)
                .where(AuditLog.needs_review.is_(True), AuditLog.review_status == "pending")
            ).scalar()
            agreed = session.execute(
                select(func.count()).select_from(AuditLog).where(AuditLog.review_status == "agreed")
            ).scalar()
            disagreed = session.execute(
                select(func.count()).select_from(AuditLog).where(AuditLog.review_status == "disagreed")
            ).scalar()
            reviewed = agreed + disagreed
            return {
                "total_messages": total_messages,
                "total_processed": total_processed,
                "pending_review": pending_review,
                "reviewed_count": reviewed,
                "model_human_agreement_rate": (agreed / reviewed) if reviewed else None,
            }


# Module-level singleton
db_service = DatabaseService()
