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
        schema_path: str = "db_schema_config.json",
        sample_rows: int = 2,
        pool_size: int = 5,
        max_overflow: int = 10,
        pool_recycle: int = 3600,
        lazy_init: bool = False,
        discover_tables: bool = False,
        view_support: bool = True,
        validate_schema: bool = False,
    ):
        """
        Initialize the database manager with connection pooling.

        Args:
            db_uri: Database connection URI
            schema_path: Path to schema configuration file (mandatory)
            sample_rows: Number of sample rows in table info
            pool_size: Number of connections to maintain in the pool (default: 5)
            max_overflow: Max connections to create beyond pool_size (default: 10)
            pool_recycle: Recycle connections after N seconds (default: 3600)
            lazy_init: If True, delay connection until first use (default: False)
            discover_tables: If True, discover all DB tables for matching (default: False)
            view_support: If True, include database views (default: True)
            validate_schema: If True, validate schema objects exist in DB (default: False)
        """
        self.db_uri = db_uri
        self.schema_path = schema_path
        self.sample_rows = sample_rows
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.discover_tables = discover_tables
        self.view_support = view_support
        self.validate_schema = validate_schema
        self._db: Optional[SQLDatabase] = None

        # Load schema config to get table names
        self.custom_schema = self.load_custom_schema(schema_path)
        self.include_tables = list(self.custom_schema.keys())
        logger.info(
            f"Loaded {len(self.include_tables)} tables from custom schema: {self.include_tables}"
        )
        logger.info(
            "Using custom schema mode: Agent will NOT read actual table metadata from database"
        )

        if not lazy_init:
            self._db = self._init_db()
            if self.validate_schema:
                self.validate_tables()

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

            # For MS SQL Server, let SQLAlchemy discover tables first, then filter
            # This avoids table name format mismatches
            if self.include_tables and self.discover_tables:
                logger.info(f"Requested tables from config: {self.include_tables}")

                # First, connect without filter to discover actual table names
                db_temp = SQLDatabase.from_uri(
                    self.db_uri,
                    sample_rows_in_table_info=0,  # Don't load samples yet
                    view_support=self.view_support,
                    engine_args=engine_args,
                )
                all_tables = db_temp.get_usable_table_names()
                logger.info(f"All available tables in database: {all_tables}")

                # Match requested tables (case-insensitive, with/without schema)
                matched_tables = []
                for requested in self.include_tables:
                    # Try exact match first
                    if requested in all_tables:
                        matched_tables.append(requested)
                        continue

                    # Try matching just the table name (ignore schema)
                    requested_name = requested.split(".")[-1]  # Get 'users' from 'dbo.users'
                    for actual in all_tables:
                        actual_name = actual.split(".")[-1]
                        if requested_name.lower() == actual_name.lower():
                            matched_tables.append(actual)
                            logger.info(f"Matched '{requested}' to '{actual}'")
                            break

                if not matched_tables:
                    logger.warning(
                        f"No tables matched! Requested: {self.include_tables}, "
                        f"Available: {all_tables}"
                    )
                    # Fall back to all tables rather than failing
                    matched_tables = list(all_tables)

                logger.info(f"Using tables: {matched_tables}")

                # Generate custom table info to prevent reading from database
                custom_table_info = self.generate_custom_table_info()
                logger.info(f"Using custom schema info for {len(custom_table_info)} tables")

                # Now create the real connection with matched table names
                # Use custom_table_info to override automatic schema reading
                db = SQLDatabase.from_uri(
                    self.db_uri,
                    include_tables=matched_tables,
                    sample_rows_in_table_info=0,  # Set to 0 since we use custom info
                    custom_table_info=custom_table_info,  # Use our custom schema
                    view_support=self.view_support,
                    engine_args=engine_args,
                )
            elif self.include_tables:
                # Use config tables directly without discovery
                logger.info(f"Using tables from config (no discovery): {self.include_tables}")
                custom_table_info = self.generate_custom_table_info()
                logger.info(f"Using custom schema info for {len(custom_table_info)} tables")

                db = SQLDatabase.from_uri(
                    self.db_uri,
                    include_tables=self.include_tables,
                    sample_rows_in_table_info=0,
                    custom_table_info=custom_table_info,
                    view_support=self.view_support,
                    engine_args=engine_args,
                )
            else:
                # No filtering requested - use all tables from custom schema
                custom_table_info = self.generate_custom_table_info()
                logger.info(f"Using custom schema info for {len(custom_table_info)} tables")

                db = SQLDatabase.from_uri(
                    self.db_uri,
                    sample_rows_in_table_info=0,  # Set to 0 since we use custom info
                    custom_table_info=custom_table_info,  # Use our custom schema
                    view_support=self.view_support,
                    engine_args=engine_args,
                )

            actual_tables = list(db.get_usable_table_names())
            logger.info(f"Database initialized with {len(actual_tables)} tables: {actual_tables}")

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

    def validate_tables(self) -> list[str]:
        """
        Validate that schema tables exist in the database.

        Logs warnings for missing tables but doesn't fail.

        Returns:
            List of missing table names
        """
        try:
            db = self.get_database()
            actual_tables = set(db.get_usable_table_names())
            missing_tables = [t for t in self.include_tables if t not in actual_tables]

            if missing_tables:
                logger.warning(
                    f"Schema tables not found in database: {missing_tables}. "
                    f"Available tables: {sorted(actual_tables)}"
                )
            else:
                logger.info(f"✅ All {len(self.include_tables)} schema tables exist in database")

            return missing_tables
        except Exception as e:
            logger.error(f"Failed to validate tables: {e}")
            return []

    def is_table_allowed(self, table_name: str) -> bool:
        """
        Check if a table is in the allowed include_tables list.

        Performs case-insensitive matching and supports both fully-qualified
        table names (e.g., 'dbo.users') and simple names (e.g., 'users').

        Args:
            table_name: Table name to check

        Returns:
            True if table is allowed, False otherwise
        """
        if not self.include_tables:
            # If no restrictions, allow all tables
            return True

        # Normalize the input table name
        normalized_input = table_name.lower().strip()
        input_simple = normalized_input.split(".")[-1]  # Get just 'users' from 'dbo.users'

        for allowed_table in self.include_tables:
            normalized_allowed = allowed_table.lower().strip()
            allowed_simple = normalized_allowed.split(".")[-1]

            # Match either full name or simple name
            if normalized_input == normalized_allowed or input_simple == allowed_simple:
                return True

        return False

    def validate_table_access(self, table_name: str) -> None:
        """
        Validate that a table is in the allowed include_tables list.

        Args:
            table_name: Table name to validate

        Raises:
            ValueError: If table is not in the allowed list
        """
        if not self.is_table_allowed(table_name):
            allowed_str = ", ".join(self.include_tables)
            raise ValueError(
                f"Access denied: Table '{table_name}' is not in the allowed tables list. "
                f"Allowed tables: [{allowed_str}]"
            )

    def _extract_table_names_from_query(self, query: str) -> list[str]:
        """
        Extract table names from a SQL query.

        This is a basic implementation that looks for common SQL patterns.
        It may not catch all edge cases but provides reasonable protection.

        Args:
            query: SQL query string

        Returns:
            List of table names found in the query
        """
        import re

        # Normalize query
        query_upper = query.upper()

        # Pattern to find table names after FROM and JOIN keywords
        # Matches: FROM table_name, JOIN schema.table_name, etc.
        patterns = [
            r"\bFROM\s+([\w.]+)",
            r"\bJOIN\s+([\w.]+)",
            r"\bINTO\s+([\w.]+)",
            r"\bUPDATE\s+([\w.]+)",
        ]

        tables = set()
        for pattern in patterns:
            matches = re.finditer(pattern, query_upper)
            for match in matches:
                table_name = match.group(1).strip()
                # Remove common SQL keywords that might be captured
                if table_name not in ["SELECT", "WHERE", "GROUP", "ORDER", "HAVING"]:
                    tables.add(table_name)

        return list(tables)

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

    def generate_custom_table_info(self) -> dict:
        """
        Generate custom_table_info dictionary for SQLDatabase.

        This creates a formatted dictionary that SQLDatabase can use instead of
        reading actual table metadata from the database.

        Returns:
            Dictionary mapping table names to formatted schema information
        """
        custom_table_info = {}

        for table_name, table_data in self.custom_schema.items():
            # Build column information
            columns_text = []
            for col_name, col_desc in table_data.get("columns", {}).items():
                columns_text.append(f"  - {col_name}: {col_desc}")

            # Build foreign key information
            fk_text = []
            for fk in table_data.get("foreign_keys", []):
                fk_text.append(f"  - {fk['column']} -> {fk['foreign_column']}")

            # Format complete table info
            info_parts = [
                f"Table: {table_name}",
                f"Description: {table_data.get('description', 'No description')}",
                "\nColumns:",
            ]
            info_parts.extend(columns_text)

            if fk_text:
                info_parts.append("\nForeign Keys:")
                info_parts.extend(fk_text)

            custom_table_info[table_name] = "\n".join(info_parts)

        return custom_table_info
