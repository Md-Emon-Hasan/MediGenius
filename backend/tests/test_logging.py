"""Tests for logging configuration"""
import logging
import os
import sys

# Add app to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))

from core.logging_config import logger, setup_logging  # noqa: E402


def test_setup_logging_creates_directory():
    """Test that setup_logging creates log directory"""
    test_log_dir = "test_logs"
    if os.path.exists(test_log_dir):
        import shutil
        shutil.rmtree(test_log_dir)

    test_logger = setup_logging(log_dir=test_log_dir)
    assert os.path.exists(test_log_dir)
    assert isinstance(test_logger, logging.Logger)

    # Cleanup
    import shutil
    shutil.rmtree(test_log_dir)


def test_logger_instance():
    """Test logger instance"""
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "medigenius"


def test_logger_has_handlers():
    """Test logger has handlers"""
    assert len(logger.handlers) > 0


def test_logger_level():
    """Test logger level"""
    assert logger.level == logging.INFO
