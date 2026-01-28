"""Tools Manager - Handles tool registration and configuration."""

import logging
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain.tools import StructuredTool

from tools import (
    calculator,
    weather,
    read_file,
    write_file,
    http_get,
    list_dir,
    system_info,
    random_joke,
)

logger = logging.getLogger(__name__)


class ToolsManager:
    """Manages tool registration and configuration."""

    def __init__(
        self,
        db: SQLDatabase,
        llm: ChatOpenAI,
        output_limit: int = 10000,
        enable_sql_tool: bool = True,
    ):
        """
        Initialize the tools manager.

        Args:
            db: Database instance for SQL tools
            llm: Language model for toolkit
            output_limit: Maximum output size for SQL queries
            enable_sql_tool: Enable or disable SQL query tool
        """
        self.db = db
        self.llm = llm
        self.output_limit = output_limit
        self.enable_sql_tool = enable_sql_tool
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
            *sql_tools,
        ]
        logger.info(f"Registered {len(tools_list)} tools " f"(utility: 8, SQL: {len(sql_tools)})")
        return tools_list

    def _get_sql_tools(self) -> list:
        """Get SQL tools with output limiting."""
        logger.debug("Creating SQL toolkit...")
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        sql_tools = sql_toolkit.get_tools()
        logger.debug(f"SQL toolkit created with {len(sql_tools)} tools")

        # Replace SQL query tool with limited version
        modified_tools: list = []
        for tool in sql_tools:
            if tool.name == "sql_db_query":
                # Create a wrapper with output limiting
                output_limit = self.output_limit
                original_tool = tool

                def limited_query_func(query: str) -> str:
                    result = original_tool.invoke(query)
                    if len(str(result)) > output_limit:
                        truncated = f"\n... (output truncated at {output_limit//1000}K chars)"
                        return str(result)[:output_limit] + truncated
                    return result

                # Create new tool with same metadata but limited function
                limited_tool = StructuredTool.from_function(
                    func=limited_query_func,
                    name=tool.name,
                    description=tool.description,
                )
                logger.debug(f"SQL query tool wrapped with {output_limit} char limit")
                modified_tools.append(limited_tool)
            else:
                modified_tools.append(tool)

        logger.debug(f"SQL tools processed: {len(modified_tools)} tools ready")
        return modified_tools

    def get_tools(self) -> list:
        """Get all registered tools."""
        return self.tools
