"""Centralised logging configuration."""
import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with a consistent format."""
    # Safely import LOG_LEVEL with a fallback so the logger
    # never causes an ImportError even during circular imports.
    try:
        from app.config import LOG_LEVEL
        level = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
    except Exception:
        level = logging.INFO

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    logger.setLevel(level)
    logger.propagate = False
    return logger