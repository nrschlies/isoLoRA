# =============================================================================
# Logging helpers for IsoLoRA
# =============================================================================
"""
Lightweight wrapper around Python `logging` **and** TensorBoard.

* `get_logger()` → configured `logging.Logger` + optional `SummaryWriter`
* Colourised console output
* File + TensorBoard logging only on rank 0 (DDP-friendly)
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

from torch.utils.tensorboard import SummaryWriter

__all__ = ["get_logger", "create_writer"]

_DATE_FMT = "%Y-%m-%dT%H:%M:%S%z"


class _ColourFormatter(logging.Formatter):
    _COL = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[41m",
    }

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        base = super().format(record)
        colour, reset = self._COL.get(record.levelno, ""), "\033[0m"
        return f"{colour}{base}{reset if colour else ''}"


def _console_handler(level: int) -> logging.Handler:
    h = logging.StreamHandler()
    h.setLevel(level)
    h.setFormatter(_ColourFormatter(
        "[%(asctime)s|%(levelname)8s|%(name)s] %(message)s", datefmt=_DATE_FMT
    ))
    return h


def _file_handler(log_dir: Path, level: int) -> logging.Handler:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    path = log_dir / f"run_{ts}.log"
    h = logging.FileHandler(path, mode="w")
    h.setLevel(level)
    h.setFormatter(logging.Formatter(
        "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
    ))
    return h


def get_logger(
    name: str = "iso_lora",
    log_dir: str | os.PathLike | None = None,
    level: int = logging.INFO,
    rank: int = 0,
) -> Tuple[logging.Logger, SummaryWriter | None]:
    """Return a configured logger (+ SummaryWriter if `log_dir` & rank 0)."""
    logger = logging.getLogger(name)

    if not logger.handlers:                       # avoid duplicates in notebooks
        logger.setLevel(level)
        logger.propagate = False
        logger.addHandler(_console_handler(level))

        if log_dir and rank == 0:
            logger.addHandler(_file_handler(Path(log_dir), level))

    writer = SummaryWriter(log_dir) if (log_dir and rank == 0) else None
    return logger, writer