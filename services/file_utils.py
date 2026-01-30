"""File utility functions for reading configuration and text files."""

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


def read_json_file(file_path: str) -> dict[str, Any]:
    """
    Read and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON content as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file contains invalid JSON
    """
    logger.debug(f"Reading JSON file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = json.load(f)
        logger.debug(f"Successfully loaded JSON from {file_path}")
        return content
    except FileNotFoundError:
        logger.error(f"JSON file not found: {file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON file {file_path}: {e}")
        raise


def read_text_file(file_path: str) -> str:
    """
    Read a text file.

    Args:
        file_path: Path to the text file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    logger.debug(f"Reading text file: {file_path}")
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        logger.debug(f"Successfully loaded text file {file_path} ({len(content)} chars)")
        return content
    except FileNotFoundError:
        logger.error(f"Text file not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {e}")
        raise
