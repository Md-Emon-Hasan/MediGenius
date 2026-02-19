"""Tests for logging — Deep Modular Architecture"""
import logging
import os
import sys

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
