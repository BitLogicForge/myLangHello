"""Tools Manager - Handles tool registration and configuration."""

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


class ToolsManager:
    """Manages tool registration and configuration."""

    def __init__(self, db: SQLDatabase, llm: ChatOpenAI, output_limit: int = 10000):
        """
        Initialize the tools manager.

        Args:
            db: Database instance for SQL tools
            llm: Language model for toolkit
            output_limit: Maximum output size for SQL queries
        """
        self.db = db
        self.llm = llm
        self.output_limit = output_limit
        self.tools = self._register_tools()

    def _register_tools(self) -> list:
        """Register and configure all tools."""
        sql_tools = self._get_sql_tools()

        return [
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

    def _get_sql_tools(self) -> list:
        """Get SQL tools with output limiting."""
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        sql_tools = sql_toolkit.get_tools()

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
                modified_tools.append(limited_tool)
            else:
                modified_tools.append(tool)

        return modified_tools

    def get_tools(self) -> list:
        """Get all registered tools."""
        return self.tools
