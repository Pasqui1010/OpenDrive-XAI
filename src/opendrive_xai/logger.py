"""Project-wide logging helper."""

import logging
from typing import Optional


def get_logger(name: Optional[str] = None, level: str = "INFO") -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str | None
        Logger name. If None, returns the root logger.
    level : str
        Logging level, default "INFO".
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        # Configure only once
        handler = logging.StreamHandler()
        fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)

    logger.setLevel(level.upper())
    return logger 