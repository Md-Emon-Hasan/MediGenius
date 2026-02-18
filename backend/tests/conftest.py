"""Test configuration and fixtures — Deep Modular Architecture"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add backend root to path so `app.*` imports work
backend_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from app.main import app  # noqa: E402


@pytest.fixture(scope="function")
def test_client():
    """Test client fixture"""
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies"""
    with patch('app.services.database_service.db_service.init_db') as mock_db, \
            patch('app.services.chat_service.chat_service.initialize_workflow'), \
            patch('app.main.process_pdf') as mock_pdf, \
            patch('app.main.get_or_create_vectorstore') as mock_vs, \
            patch('app.services.database_service.db_service.save_message') as mock_save:

        mock_vs.return_value = MagicMock()

        mock_app_instance = MagicMock()
        mock_app_instance.ainvoke = AsyncMock(return_value={
            "generation": "Test response from AI",
            "source": "Test Source",
            "llm_success": True
        })
        mock_app_instance.invoke.return_value = {
            "generation": "Test response from AI",
            "source": "Test Source",
            "llm_success": True
        }

        with patch('app.services.chat_service.chat_service.workflow_app', mock_app_instance):
            yield {
                "db": mock_db,
                "pdf": mock_pdf,
                "vector_store": mock_vs,
                "workflow_app": mock_app_instance,
                "save_message": mock_save,
            }


@pytest.fixture
def mock_session_middleware():
    """Mock session middleware behavior"""
    pass
