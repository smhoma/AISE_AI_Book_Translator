"""Central logging configuration for command-line and Gradio entrypoints."""

import logging
import os
import sys


DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DEFAULT_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: str | int | None = None) -> None:
    """Configure project logging with DEBUG as the default application level."""
    log_level = level or os.getenv("LOG_LEVEL", "DEBUG")
    if isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.DEBUG)

    logging.basicConfig(
        level=log_level,
        format=DEFAULT_LOG_FORMAT,
        datefmt=DEFAULT_DATE_FORMAT,
        stream=sys.stdout,
        force=True,
    )

    for noisy_logger in ["asyncio", "httpx", "httpcore", "urllib3", "PIL", "gradio", "openai", "matplotlib"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
