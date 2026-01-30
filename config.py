"""Configuration manager - singleton for loading and accessing config."""

import logging
from typing import Optional, Any
from services.file_utils import read_json_file

logger = logging.getLogger(__name__)


class Config:
    """Singleton configuration manager that loads config once and caches it."""

    _instance: Optional["Config"] = None
    _config: Optional[dict[str, Any]] = None
    _config_path: str = "config.json"

    def __new__(cls, config_path: str = "config.json"):
        """Create singleton instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._config_path = config_path
        return cls._instance

    def __init__(self, config_path: str = "config.json"):
        """Initialize config manager (only loads config once)."""
        # Only load config once
        if Config._config is None:
            Config._config_path = config_path
            self._load_config()

    def _load_config(self) -> None:
        """Load configuration from file (called only once)."""
        logger.info(f"Loading configuration from {Config._config_path}")
        Config._config = read_json_file(Config._config_path)
        logger.info("Configuration loaded successfully")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (supports nested keys with dot notation, e.g., 'azure.model')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        if Config._config is None:
            self._load_config()

        # Support nested keys with dot notation
        keys = key.split(".")
        value = Config._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_all(self) -> dict[str, Any]:
        """
        Get the entire configuration dictionary.

        Returns:
            Complete configuration
        """
        if Config._config is None:
            self._load_config()
        return Config._config.copy() if Config._config else {}

    @classmethod
    def reload(cls) -> None:
        """Force reload of configuration from file."""
        logger.info("Reloading configuration")
        cls._config = None
        if cls._instance:
            cls._instance._load_config()

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton instance (useful for testing)."""
        cls._instance = None
        cls._config = None
