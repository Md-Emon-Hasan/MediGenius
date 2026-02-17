"""Database service layer for MediGenius using SQLAlchemy"""
from typing import Dict, List, Optional

from core.logging_config import logger
from models.message import Base, Message
from sqlalchemy import create_engine, delete, desc, func, select
from sqlalchemy.orm import Session, sessionmaker


class DatabaseService:
    """Service class for database operations using SQLAlchemy"""

    def __init__(self, db_path: str = "data/medigenius.db"):
        """Initialize database service"""
        # Ensure data directory exists if provided
        import os
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self.db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(self.db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        logger.info(f"DatabaseService initialized with SQLAlchemy: {self.db_url}")
        self.init_db()

    def init_db(self):
        """Initialize database tables"""
        logger.info("Initializing database tables using SQLAlchemy...")
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a new database session"""
        return self.SessionLocal()

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        source: Optional[str] = None
    ) -> None:
        """Save a message to the database"""
        logger.debug(f"Saving {role} message for session {session_id[:8]}...")
        with self.get_session() as session:
            db_message = Message(
                session_id=session_id,
                role=role,
                content=content,
                source=source
            )
            session.add(db_message)
            session.commit()
        logger.debug("Message saved successfully")

    def get_chat_history(self, session_id: str) -> List[Dict]:
        """Get chat history for a session"""
        logger.debug(f"Retrieving chat history for session {session_id[:8]}...")
        with self.get_session() as session:
            stmt = select(Message).where(Message.session_id == session_id).order_by(Message.timestamp)
            results = session.execute(stmt).scalars().all()
            messages = [msg.to_dict() for msg in results]

        logger.debug(f"Retrieved {len(messages)} messages")
        return messages

    def get_all_sessions(self) -> List[Dict]:
        """Get all chat sessions with preview"""
        logger.debug("Retrieving all sessions...")
        with self.get_session() as session:
            # Subquery to find the latest timestamp for each session
            latest_msg_sub = select(
                Message.session_id,
                func.max(Message.timestamp).label("max_ts")
            ).where(Message.role == 'user').group_by(Message.session_id).subquery()

            # Join with Message table to get the content of those latest messages/sessions
            stmt = select(
                Message.session_id,
                Message.content,
                Message.timestamp
            ).join(
                latest_msg_sub,
                (Message.session_id == latest_msg_sub.c.session_id) & (Message.timestamp == latest_msg_sub.c.max_ts)
            ).order_by(desc(Message.timestamp))

            results = session.execute(stmt).all()

            sessions = []
            for row in results:
                sessions.append({
                    'session_id': row[0],
                    'preview': row[1][:50] + '...' if len(row[1]) > 50 else row[1],
                    'last_active': row[2].isoformat() if row[2] else None
                })

        return sessions

    def delete_session(self, session_id: str) -> None:
        """Delete all messages for a session"""
        logger.info(f"Deleting session {session_id[:8]}...")
        with self.get_session() as session:
            stmt = delete(Message).where(Message.session_id == session_id)
            session.execute(stmt)
            session.commit()
        logger.info("Session deleted successfully")


# Global database service instance
db_service = DatabaseService()
