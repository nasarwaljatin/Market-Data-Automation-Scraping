"""
utils/logger.py
---------------
Structured logger used across all pipeline stages.
Logs to both console (INFO) and file (DEBUG) with timestamps.

Windows notes:
  - stdout is reconfigured to UTF-8 so Unicode arrows/box-drawing
    characters don't crash on cp1252 consoles.
  - Uses a plain FileHandler (not RotatingFileHandler) to avoid
    WinError 32 log-rotation permission conflicts.
"""

import logging
import os
import sys

from utils.config import LOG_FILE


def _utf8_stdout_stream():
    """Return a UTF-8-safe stdout stream for Windows consoles."""
    try:
        # Python 3.7+: reconfigure if supported
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        return sys.stdout
    except Exception:
        import io
        return io.TextIOWrapper(
            sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True
        )


def get_logger(name: str, level: int = logging.DEBUG) -> logging.Logger:
    """
    Return a named logger configured with:
      - StreamHandler  -> stdout (UTF-8), level INFO
      - FileHandler    -> logs/pipeline.log (UTF-8, append), level DEBUG

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

    # ── Console handler (UTF-8 safe) ─────────────────────────
    ch = logging.StreamHandler(_utf8_stdout_stream())
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # ── File handler (plain append, UTF-8) ───────────────────
    # Using FileHandler instead of RotatingFileHandler to avoid
    # WinError 32 (file locked) during rotation on Windows.
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    fh = logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8", delay=False)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger

