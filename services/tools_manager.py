"""Tools Manager - Handles tool registration and configuration."""

import logging
from typing import Optional
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import StructuredTool

from tools import (
    calculator,
    weather,
    read_file,
    write_file,
    http_get,
    list_dir,
    system_info,
    random_joke,
    current_date,
    loan_calculator,
    currency_converter,
    joke_format,
)

logger = logging.getLogger(__name__)


class ToolsManager:
    """Manages tool registration and configuration."""

    def __init__(
        self,
        db: Optional[SQLDatabase],
        llm: BaseChatModel,
        output_limit: int = 10000,
        enable_sql_tool: bool = True,
        db_manager=None,
    ):
        """
        Initialize the tools manager.

        Args:
            db: Database instance for SQL tools (None if SQL tools disabled)
            llm: Language model for toolkit
            output_limit: Maximum output size for SQL queries
            enable_sql_tool: Enable or disable SQL query tool
            db_manager: DatabaseManager instance for table access validation
        """
        self.db = db
        self.llm = llm
        self.output_limit = output_limit
        self.enable_sql_tool = enable_sql_tool
        self.db_manager = db_manager
        sql_status = "enabled" if enable_sql_tool else "disabled"
        logger.debug(
            f"ToolsManager initializing (SQL tools: {sql_status}, " f"output_limit: {output_limit})"
        )
        self.tools = self._register_tools()
        logger.info(f"ToolsManager initialized with {len(self.tools)} tools")

    def _register_tools(self) -> list:
        """Register and configure all tools."""
        logger.debug("Registering tools...")
        sql_tools = self._get_sql_tools() if self.enable_sql_tool else []

        tools_list = [
            calculator,
            weather,
            read_file,
            write_file,
            http_get,
            list_dir,
            system_info,
            random_joke,
            current_date,
            loan_calculator,
            currency_converter,
            joke_format,
            *sql_tools,
        ]
        logger.info(f"Registered {len(tools_list)} tools " f"(utility: 8, SQL: {len(sql_tools)})")
        return tools_list

    def _get_sql_tools(self) -> list:
        """Get SQL tools with output limiting and table access control."""
        if self.db is None:
            logger.warning("Database is None, cannot create SQL tools")
            return []

        logger.debug("Creating SQL toolkit...")
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        sql_tools = sql_toolkit.get_tools()
        logger.debug(f"SQL toolkit created with {len(sql_tools)} tools")

        # Replace SQL tools with protected versions
        modified_tools: list = []
        for tool in sql_tools:
            if tool.name == "sql_db_query":
                # Create wrapper with output limiting, read-only enforcement,
                # and table access control
                output_limit = self.output_limit
                original_tool = tool
                db_manager = self.db_manager

                def limited_query_func(query: str) -> str:
                    # Block DDL/DML commands
                    query_upper = query.strip().upper()
                    dangerous_keywords = [
                        "CREATE",
                        "DROP",
                        "INSERT",
                        "UPDATE",
                        "DELETE",
                        "ALTER",
                        "TRUNCATE",
                    ]

                    for keyword in dangerous_keywords:
                        if query_upper.startswith(keyword):
                            logger.warning(f"Blocked dangerous SQL command: {keyword}")
                            return (
                                f"Error: {keyword} commands are not allowed. "
                                "Only SELECT queries permitted."
                            )

                    # Validate table access if db_manager is available
                    if db_manager:
                        try:
                            # Extract table names from query (basic parsing)
                            tables = db_manager._extract_table_names_from_query(query)
                            for table in tables:
                                db_manager.validate_table_access(table)
                        except ValueError as e:
                            logger.warning(f"Table access denied: {e}")
                            return str(e)
                        except Exception as e:
                            logger.error(f"Error validating table access: {e}")
                            # Continue with query if validation fails unexpectedly

                    result = original_tool.invoke(query)
                    if len(str(result)) > output_limit:
                        truncated = f"\n... (output truncated at {output_limit//1000}K chars)"
                        return str(result)[:output_limit] + truncated
                    return result

                limited_tool = StructuredTool.from_function(
                    func=limited_query_func,
                    name=tool.name,
                    description=(
                        tool.description
                        + " (Read-only: SELECT queries only, restricted to allowed tables)"
                    ),
                )
                logger.debug(
                    f"SQL query tool wrapped with {output_limit} char limit, "
                    "read-only protection, and table access control"
                )
                modified_tools.append(limited_tool)

            elif tool.name == "sql_db_schema":
                # Wrap schema tool to only show allowed tables
                original_tool = tool
                db_manager = self.db_manager

                def restricted_schema_func(table_names: str) -> str:
                    # Validate each requested table
                    if db_manager:
                        tables_list = [t.strip() for t in table_names.split(",")]
                        for table in tables_list:
                            try:
                                db_manager.validate_table_access(table)
                            except ValueError as e:
                                logger.warning(f"Schema access denied for table: {e}")
                                return str(e)

                    return original_tool.invoke(table_names)

                restricted_tool = StructuredTool.from_function(
                    func=restricted_schema_func,
                    name=tool.name,
                    description=tool.description + " (Restricted to allowed tables only)",
                )
                logger.debug("SQL schema tool wrapped with table access control")
                modified_tools.append(restricted_tool)

            elif tool.name == "sql_db_list_tables":
                # Wrap list tables tool to only show allowed tables
                db_manager = self.db_manager

                def restricted_list_func(tool_input: str = "") -> str:
                    if db_manager and db_manager.include_tables:
                        allowed = ", ".join(db_manager.include_tables)
                        return f"Allowed tables (restricted by configuration): {allowed}"
                    else:
                        # If no restrictions, use original functionality
                        return original_tool.invoke(tool_input)

                original_tool = tool
                restricted_tool = StructuredTool.from_function(
                    func=restricted_list_func,
                    name=tool.name,
                    description=(
                        "List available database tables " "(restricted to allowed tables only)"
                    ),
                )
                logger.debug("SQL list tables tool wrapped with table access control")
                modified_tools.append(restricted_tool)

            else:
                # For other tools (like sql_db_query_checker), pass through unchanged
                modified_tools.append(tool)

        logger.debug(
            f"SQL tools processed: {len(modified_tools)} tools ready " "with table access control"
        )
        return modified_tools

    def get_tools(self) -> list:
        """Get all registered tools."""
        return self.tools
