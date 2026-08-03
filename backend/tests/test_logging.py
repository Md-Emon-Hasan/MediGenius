"""Tests for logging — Deep Modular Architecture"""
import logging
import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.core.logging_config import logger, setup_logging  # noqa: E402


def test_setup_logging_creates_directory():
    # This test is now bypassed or updated because setup_logging skips dir creation in tests.
    # We verify that in test mode, no dir is created even if requested.
    test_log_dir = "test_logs_should_not_exist"
    if os.path.exists(test_log_dir):
        import shutil
        shutil.rmtree(test_log_dir)

    setup_logging(log_dir=test_log_dir)
    # With the new zero-log policy, this directory should NOT be created during tests
    assert not os.path.exists(test_log_dir)


def test_logger_instance():
    assert logger is not None
    assert isinstance(logger, logging.Logger)
    assert logger.name == "medigenius"


def test_logger_has_handlers():
    assert len(logger.handlers) > 0


def test_logger_level():
    # In pytest env, level is set to DEBUG
    assert logger.level == logging.DEBUG


def test_setup_logging_production_branch_creates_file_handler():
    medigenius_logger = logging.getLogger("medigenius")
    saved_handlers = medigenius_logger.handlers[:]
    medigenius_logger.handlers = []
    try:
        with patch("os.path.exists", return_value=False), \
             patch("os.makedirs") as mock_makedirs, \
             patch("app.core.logging_config.RotatingFileHandler") as mock_handler_cls:
            mock_handler_cls.return_value = MagicMock()
            result = setup_logging(log_dir="fake_prod_logs", is_testing=False)
            mock_makedirs.assert_called_once_with("fake_prod_logs")
            mock_handler_cls.assert_called_once()
            assert result.level == logging.INFO
    finally:
        medigenius_logger.handlers = saved_handlers


def test_setup_logging_production_branch_skips_makedirs_if_dir_exists():
    medigenius_logger = logging.getLogger("medigenius")
    saved_handlers = medigenius_logger.handlers[:]
    medigenius_logger.handlers = []
    try:
        with patch("os.path.exists", return_value=True), \
             patch("os.makedirs") as mock_makedirs, \
             patch("app.core.logging_config.RotatingFileHandler") as mock_handler_cls:
            mock_handler_cls.return_value = MagicMock()
            setup_logging(log_dir="fake_prod_logs", is_testing=False)
            mock_makedirs.assert_not_called()
    finally:
        medigenius_logger.handlers = saved_handlers
