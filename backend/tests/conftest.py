"""Test configuration and fixtures"""
import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Add app directory to path
app_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app'))
if app_path not in sys.path:
    sys.path.insert(0, app_path)

from main import app  # noqa: E402


@pytest.fixture(scope="function")
def test_client():
    """Test client fixture"""
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies"""
    with patch('services.database_service.db_service.init_db') as mock_db, \
            patch('services.chat_service.chat_service.initialize_workflow'), \
            patch('tools.pdf_loader.process_pdf') as mock_pdf, \
            patch('tools.vector_store.get_or_create_vectorstore') as mock_vs:

        # Setup mock behaviors
        mock_vs.return_value = MagicMock()

        # Mock the workflow app
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

        # Patch chat_service.workflow_app
        with patch('services.chat_service.chat_service.workflow_app', mock_app_instance):
            yield {
                "db": mock_db,
                "pdf": mock_pdf,
                "vector_store": mock_vs,
                "workflow_app": mock_app_instance
            }


@pytest.fixture
def mock_session_middleware():
    """Mock session middleware behavior"""
    pass
