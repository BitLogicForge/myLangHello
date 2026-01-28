"""Database Manager - Handles database connection and schema configuration."""

import json
from typing import Optional
from langchain_community.utilities import SQLDatabase
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and schema information with connection pooling."""

    def __init__(
        self,
        db_uri: str,
        include_tables: Optional[list] = None,
        sample_rows: int = 2,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        lazy_init: bool = False,
    ):
        """
        Initialize the database manager with connection pooling.

        Args:
            db_uri: Database connection URI
            include_tables: List of tables to include
            sample_rows: Number of sample rows in table info
            pool_size: Number of connections to maintain in the pool (default: 5)
            max_overflow: Max connections to create beyond pool_size (default: 10)
            pool_recycle: Recycle connections after N seconds (default: 3600)
            lazy_init: If True, delay connection until first use (default: False)
        """
        self.db_uri = db_uri
        self.include_tables = include_tables or []
        self.sample_rows = sample_rows
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self._db: Optional[SQLDatabase] = None

        if not lazy_init:
            self._db = self._init_db()

    def _init_db(self) -> SQLDatabase:
        """Initialize database connection with pooling configuration."""
        try:
            # Add connection pool parameters to URI if using SQLAlchemy
            engine_args = {
                "pool_size": self.pool_size,
                "max_overflow": self.max_overflow,
                "pool_recycle": self.pool_recycle,
                "pool_pre_ping": True,  # Verify connections before using
            }

            db = SQLDatabase.from_uri(
                self.db_uri,
                include_tables=self.include_tables,
                sample_rows_in_table_info=self.sample_rows,
                engine_args=engine_args,
            )
            logger.info(f"Database connection initialized with pool_size={self.pool_size}")
            return db
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    def get_database(self) -> SQLDatabase:
        """Get the database instance, initializing if needed."""
        if self._db is None:
            self._db = self._init_db()
        return self._db

    def reconnect(self) -> None:
        """Force reconnection to database."""
        logger.info("Reconnecting to database...")
        self.close()
        self._db = self._init_db()

    def close(self) -> None:
        """Close database connection and dispose of connection pool."""
        if self._db is not None:
            try:
                self._db._engine.dispose()
                logger.info("Database connection pool disposed")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
            finally:
                self._db = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        db = self.get_database()
        try:
            yield db
        except Exception as e:
            logger.error(f"Database operation error: {e}")
            raise

    def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            db = self.get_database()
            # Execute a simple query to verify connection
            db.run("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

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
