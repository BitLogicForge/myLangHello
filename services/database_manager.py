"""Database Manager - Handles database connection and schema configuration."""

import json
from typing import Optional
from langchain_community.utilities import SQLDatabase


class DatabaseManager:
    """Manages database connections and schema information."""

    def __init__(self, db_uri: str, include_tables: Optional[list] = None, sample_rows: int = 2):
        """
        Initialize the database manager.

        Args:
            db_uri: Database connection URI
            include_tables: List of tables to include
            sample_rows: Number of sample rows in table info
        """
        self.db_uri = db_uri
        self.include_tables = include_tables or []
        self.sample_rows = sample_rows
        self.db = self._init_db()

    def _init_db(self) -> SQLDatabase:
        """Initialize database connection."""
        return SQLDatabase.from_uri(
            self.db_uri,
            include_tables=self.include_tables,
            sample_rows_in_table_info=self.sample_rows,
        )

    def get_database(self) -> SQLDatabase:
        """Get the database instance."""
        return self.db

    @staticmethod
    def load_custom_schema(schema_path: str = "db_schema_config.json") -> dict:
        """
        Load custom schema information from JSON file.

        Args:
            schema_path: Path to schema configuration file

        Returns:
            Dictionary containing schema information
        """
        with open(schema_path, "r", encoding="utf-8") as f:
            return json.load(f)
