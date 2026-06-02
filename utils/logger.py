"""
utils/logger.py
---------------
Structured logger used across all pipeline stages.
Logs to both console (INFO) and file (DEBUG) with timestamps.
"""

import logging
import os
import sys
from logging.handlers import RotatingFileHandler

from utils.config import LOG_FILE


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger configured with:
      - StreamHandler  → stdout, level INFO
      - RotatingFileHandler → logs/pipeline.log, level DEBUG (max 5 MB × 3 backups)

    Parameters
    ----------
    name : str
        Logger name, typically __name__ of the calling module.
    level : int
        Root level for the logger (default DEBUG).

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Already configured — return as-is to avoid duplicate handlers
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (rotating) ───────────────────────────────
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fh = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
