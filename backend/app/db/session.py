"""Database session management"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import os

# Default database path
DEFAULT_DB_PATH = "database/medigenius.db"

def get_engine(db_path: str = DEFAULT_DB_PATH):
    """Create and return SQLAlchemy engine"""
    # Ensure directory exists
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)
    
    db_url = f"sqlite:///{db_path}"
    return create_engine(db_url, connect_args={"check_same_thread": False})

def get_session_factory(engine):
    """Create and return session factory"""
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Default session components
engine = get_engine()
SessionLocal = get_session_factory(engine)
