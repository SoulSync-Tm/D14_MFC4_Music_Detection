"""app/utils/logging.py — Centralised logger factory."""

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a logger that writes to stdout with a standard timestamp format."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
