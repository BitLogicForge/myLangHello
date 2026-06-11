"""Configuration manager - singleton for loading and accessing config."""

import logging
from typing import Any, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from utils import read_json_file

logger = logging.getLogger(__name__)


class AppSettings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # OpenAI Configuration
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_organization: Optional[str] = Field(default=None, alias="OPENAI_ORGANIZATION")
    openai_base_url: Optional[str] = Field(default=None, alias="OPENAI_BASE_URL")

    # Database Configuration
    db_enabled: bool = Field(default=False, alias="DB_ENABLED")
    db_host: Optional[str] = Field(default=None, alias="DB_HOST")
    db_name: Optional[str] = Field(default=None, alias="DB_NAME")
    db_username: Optional[str] = Field(default=None, alias="DB_USERNAME")
    db_password: Optional[str] = Field(default=None, alias="DB_PASSWORD")
    db_use_windows_auth: bool = Field(default=False, alias="DB_USE_WINDOWS_AUTH")
    db_driver: str = Field(default="ODBC Driver 17 for SQL Server", alias="DB_DRIVER")

    # Azure OpenAI Configuration
    azure_openai_api_key: Optional[str] = Field(default=None, alias="AZURE_OPENAI_API_KEY")
    azure_openai_endpoint: Optional[str] = Field(default=None, alias="AZURE_OPENAI_ENDPOINT")
    azure_openai_api_version: str = Field(default="2024-02-15-preview", alias="AZURE_OPENAI_API_VERSION")
    azure_openai_deployment_name: Optional[str] = Field(default=None, alias="AZURE_OPENAI_DEPLOYMENT_NAME")

    # LM Studio Configuration
    lmstudio_base_url: str = Field(default="http://localhost:1234/v1", alias="LMSTUDIO_BASE_URL")
    lmstudio_api_key: str = Field(default="lm-studio", alias="LMSTUDIO_API_KEY")

    # Ollama Configuration
    ollama_base_url: Optional[str] = Field(default=None, alias="OLLAMA_BASE_URL")
    ollama_api_key: Optional[str] = Field(default=None, alias="OLLAMA_API_KEY")

    # LangChain Tracing
    langchain_tracing_v2: bool = Field(default=False, alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: Optional[str] = Field(default=None, alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="default", alias="LANGCHAIN_PROJECT")

    # Logging Configuration
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Server Configuration
    port: int = Field(default=8000, alias="PORT")


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


# Global settings instance loaded from environment variables and .env file
settings = AppSettings()
