"""
Structured logging configuration using Loguru.
Outputs JSON-formatted logs for production, human-readable for development.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from config.settings import LOG_DIR


def setup_logging(
    level: str = "INFO",
    json_format: bool = False,
    rotation: str = "50 MB",
    retention: str = "30 days",
) -> None:
    """
    Configure application-wide logging.

    Args:
        level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        json_format: If True, output structured JSON logs (for production)
        rotation: Log file rotation size
        retention: How long to keep old log files
    """
    # Remove default handler
    logger.remove()

    # Console handler — always human-readable
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
        backtrace=True,
        diagnose=True,
    )

    # File handler — JSON for production, text for development
    if json_format:
        logger.add(
            LOG_DIR / "flood_system_{time:YYYY-MM-DD}.jsonl",
            level=level,
            format="{message}",
            serialize=True,  # JSON output
            rotation=rotation,
            retention=retention,
            compression="gz",
            enqueue=True,  # Thread-safe
        )
    else:
        logger.add(
            LOG_DIR / "flood_system_{time:YYYY-MM-DD}.log",
            level=level,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
                "{name}:{function}:{line} | {message}"
            ),
            rotation=rotation,
            retention=retention,
            compression="gz",
            enqueue=True,
        )

    # Separate error log
    logger.add(
        LOG_DIR / "errors_{time:YYYY-MM-DD}.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
        rotation=rotation,
        retention=retention,
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    logger.info(f"Logging initialized | level={level} | json={json_format} | dir={LOG_DIR}")
