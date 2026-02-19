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
from app.services import chat_service, db_service  # noqa: E402


@pytest.fixture(scope="function")
def test_client():
    """Test client fixture"""
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock all external dependencies using object-based patching to avoid CI import issues"""
    # 1. Create a workspace for all mocks
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

    # 2. Patch using patch.object on the imported singletons
    with patch.object(db_service, 'init_db') as mock_db, \
         patch.object(chat_service, 'initialize_workflow'), \
         patch.object(chat_service, 'workflow_app', mock_app_instance), \
         patch.object(db_service, 'save_message') as mock_save, \
         patch('app.main.process_pdf') as mock_pdf, \
         patch('app.main.get_or_create_vectorstore') as mock_vs:

        mock_vs.return_value = MagicMock()

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
