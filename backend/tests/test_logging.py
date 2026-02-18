"""Tests for logging — Deep Modular Architecture"""
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.logging_config import logger, setup_logging  # noqa: E402


def test_setup_logging_creates_directory():
    test_log_dir = "test_logs"
    if os.path.exists(test_log_dir):
        import shutil
        shutil.rmtree(test_log_dir)

    test_logger = setup_logging(log_dir=test_log_dir)
    assert os.path.exists(test_log_dir)
    assert isinstance(test_logger, logging.Logger)

    import shutil
    shutil.rmtree(test_log_dir)


def test_logger_instance():
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "medigenius"


def test_logger_has_handlers():
    assert len(logger.handlers) > 0


def test_logger_level():
    assert logger.level == logging.INFO
