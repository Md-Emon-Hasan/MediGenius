"""
MediGenius — models/message.py
SQLAlchemy ORM model for chat messages.
"""

from datetime import datetime
from typing import Dict

from sqlalchemy import Column, DateTime, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Message(Base):
    """Persisted chat message (user or assistant turn)."""

    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(255), nullable=False, index=True)
    role = Column(String(50), nullable=False)          # "user" | "assistant"
    content = Column(Text, nullable=False)
    source = Column(String(255), nullable=True)        # e.g. "Wikipedia Medical Information"
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "source": self.source,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
