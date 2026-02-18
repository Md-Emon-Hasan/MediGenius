"""
MediGenius — db/session.py
SQLAlchemy engine and session factory.
"""

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import CHAT_DB_PATH
from app.core.logging_config import logger


def get_engine(db_path: str = CHAT_DB_PATH):
    """Create and return a SQLAlchemy engine for the given SQLite path."""
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    logger.debug("Database engine created at %s", db_path)
    return create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )


def get_session_factory(engine):
    """Return a sessionmaker bound to the given engine."""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Module-level singletons
engine = get_engine()
SessionLocal = get_session_factory(engine)
