"""
MediGenius — core/logging_config.py
Rotating file + console logging setup.
"""

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

from app.core.config import LOG_DIR


def setup_logging(log_dir: str = LOG_DIR) -> logging.Logger:
    """Setup rotating file + console logging. Returns the 'medigenius' logger."""
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    _logger = logging.getLogger("medigenius")
    _logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-import
    if _logger.handlers:
        return _logger

    log_file = os.path.join(
        log_dir, f"medigenius_{datetime.now().strftime('%Y%m%d')}.log"
    )
    file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    _logger.addHandler(file_handler)
    _logger.addHandler(console_handler)
    return _logger


# Module-level singleton
logger = setup_logging()
