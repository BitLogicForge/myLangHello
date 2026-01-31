"""Agent Factory - Creates and configures the agent using LangGraph."""

import logging
import os
from typing import Any, Optional

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


# Load environment variables from .env file
load_dotenv()


class AgentFactory:
    """Factory for creating and configuring LangGraph ReAct agents."""

    def __init__(
        self,
        llm: BaseChatModel,
        tools: list,
        system_prompt: str,
        checkpointer: Optional[Any] = None,
    ):
        """
        Initialize the agent factory with LangGraph.

        Args:
            llm: Language model instance
            tools: List of tools for the agent
            system_prompt: System prompt text to guide agent behavior
            checkpointer: Optional memory checkpointer for conversation persistence
        """
        self.llm = llm
        self.tools = tools
        self.system_prompt = system_prompt
        self.checkpointer = checkpointer

        logger.info(f"AgentFactory initialized with {len(tools)} tools using LangGraph ReAct agent")

    def create_db_agent(self):
        """
        Create a LangGraph agent with a toolkit.

        Returns:
            LangGraph agent graph that can be invoked
        """
        logger.debug("Creating LangGraph toolkit agent...")

        # agent = create_agent(
        #     model=self.llm,
        #     tools=toolkit.get_tools(),
        #     system_prompt=self.system_prompt,
        #     checkpointer=self.checkpointer,
        #     verbose=verbose,
        # )

        db_host = os.getenv("DB_HOST")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USERNAME")
        db_password = os.getenv("DB_PASSWORD")
        db_driver = os.getenv("DB_DRIVER")

        conn_str = (
            f"mssql+pyodbc://{db_user}:{db_password}@"
            f"{db_host}/{db_name}?driver={db_driver}&TrustServerCertificate=yes"
        )

        # Create SQL toolkit with view support enabled
        toolkit = SQLDatabaseToolkit(
            db=SQLDatabase.from_uri(
                database_uri=conn_str,
                view_support=True,  # Enable querying database views
                include_tables=["nbp_countries_view"],
            ),
            llm=self.llm,
        )

        # Combine SQL toolkit tools with custom tools
        all_tools = toolkit.get_tools() + self.tools

        # Use create_agent to support custom tools
        agent = create_agent(
            model=self.llm,
            tools=all_tools,
            system_prompt=self.system_prompt,
            checkpointer=self.checkpointer,
        )

        return agent
