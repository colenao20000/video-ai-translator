"""Structured Logging Configuration (Tuyệt đối không log secret API keys)."""

import os
import logging
from logging.handlers import RotatingFileHandler


def setup_logging(log_dir: str = "~/.vtranslator/logs") -> logging.Logger:
    expanded_dir = os.path.expanduser(log_dir)
    os.makedirs(expanded_dir, exist_ok=True)
    log_file = os.path.join(expanded_dir, "app.log")

    logger = logging.getLogger("vtranslator")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d]: %(message)s"
        )

        file_handler = RotatingFileHandler(log_file, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
