"""
MediGenius — models/audit_log.py
Safety/performance audit trail. Metadata only — never the raw question or answer text.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from app.models.message import Base


class AuditLog(Base):
    """One row per processed chat message, for safety tuning and latency tracking."""

    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    safety_category = Column(String(50), nullable=True)
    refused_topic = Column(String(50), nullable=True)
    figures_removed_count = Column(Integer, default=0)
    source = Column(String(255), nullable=True)
    model_used = Column(String(100), nullable=True)
    model_fallback = Column(Boolean, default=False)
    verification_risk = Column(String(20), nullable=True)
    latency_ms = Column(Float, nullable=True)
    degraded = Column(Boolean, default=False)
    cache_hit = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Phase 8 — added via database_service._migrate_audit_log_columns for pre-existing rows
    needs_review = Column(Boolean, default=False)
    review_status = Column(String(20), default="pending")
    human_verdict = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "session_id": self.session_id,
            "safety_category": self.safety_category,
            "refused_topic": self.refused_topic,
            "figures_removed_count": self.figures_removed_count,
            "source": self.source,
            "model_used": self.model_used,
            "model_fallback": self.model_fallback,
            "verification_risk": self.verification_risk,
            "latency_ms": self.latency_ms,
            "degraded": self.degraded,
            "cache_hit": self.cache_hit,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "needs_review": self.needs_review,
            "review_status": self.review_status,
            "human_verdict": self.human_verdict,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
        }
