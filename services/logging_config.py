"""Centralized logging configuration with colorful output."""

import logging
import colorlog


def setup_logging():
    """Configure colorful logging for the application."""
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
    logging.basicConfig(level=logging.INFO, handlers=[console_handler])
