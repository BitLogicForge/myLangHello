"""Centralized logging configuration with colorful output."""

import logging
from typing import Optional

import colorlog

# Suppress verbose third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


def setup_logging(debug: bool = False, level: Optional[int] = None) -> None:
    """Configure colorful logging for the application."""
    root_level = level if level is not None else (logging.DEBUG if debug else logging.INFO)

    # Create console handler with colors
    console_handler = colorlog.StreamHandler()
    console_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
            style="%",
        )
    )

    # Configure root logger
    logging.basicConfig(level=root_level, handlers=[console_handler], force=True)
