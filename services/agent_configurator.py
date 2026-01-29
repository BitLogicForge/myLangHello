"""Agent configuration and initialization service."""

import os
import logging
from typing import Optional, Dict, Any

from .database_manager import DatabaseManager
from .tools_manager import ToolsManager
from .prompt_builder import PromptBuilder
from .agent_factory import AgentFactory
from .llm_factory import LLMFactory

logger = logging.getLogger(__name__)


class AgentConfigurator:
    """Handles agent initialization and component setup."""

    def __init__(
        self,
        db_uri: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        schema_path: str = "db_schema_config.json",
        enable_sql_tool: bool = True,
        llm_config_path: str = "llm_config.json",
        llm_provider: Optional[str] = None,
        system_prompt_path: str = "messages/system_prompt.txt",
        sample_rows: int = 2,
        pool_size: int = 10,
        max_overflow: int = 15,
        pool_recycle: int = 1800,
        output_limit: int = 10000,
        verbose: bool = True,
        max_iterations: int = 10,
        max_execution_time: int = 60,
    ):
        """
        Initialize the agent configurator.

        Args:
            db_uri: Database connection URI (MSSQL format).
                   If None, reads from DATABASE_URI env var.
            model: LLM model name
            temperature: LLM temperature setting
            schema_path: Path to database schema configuration (mandatory for SQL tools)
            enable_sql_tool: Enable or disable SQL query tool
            llm_config_path: Path to LLM configuration file
            llm_provider: Override LLM provider (azure or openai)
            system_prompt_path: Path to system prompt file
            sample_rows: Number of sample rows for database tables
            pool_size: Database connection pool size
            max_overflow: Maximum overflow connections in pool
            pool_recycle: Connection recycle time in seconds
            output_limit: Output limit for tools
            verbose: Enable verbose logging for agent
            max_iterations: Maximum agent iterations
            max_execution_time: Maximum execution time in seconds
        """
        self.db_uri = db_uri
        self.model = model
        self.temperature = temperature
        self.schema_path = schema_path
        self.enable_sql_tool = enable_sql_tool
        self.llm_config_path = llm_config_path
        self.llm_provider = llm_provider
        self.system_prompt_path = system_prompt_path
        self.sample_rows = sample_rows
        self.pool_size = pool_size
        self.max_overflow = max_overflow
        self.pool_recycle = pool_recycle
        self.output_limit = output_limit
        self.verbose = verbose
        self.max_iterations = max_iterations
        self.max_execution_time = max_execution_time

        # Component storage
        self.llm: Optional[Any] = None
        self.db_manager: Optional[DatabaseManager] = None
        self.custom_schema: Dict = {}
        self.tools_manager: Optional[ToolsManager] = None
        self.prompt_builder: Optional[PromptBuilder] = None
        self.prompt: Optional[Any] = None
        self.agent_factory: Optional[AgentFactory] = None

    def setup_llm(self) -> Any:
        """
        Initialize the LLM component.

        Returns:
            Initialized LLM instance
        """
        provider_info = self.llm_provider or "from config"
        logger.info(f"Initializing LLM: {self.model} (provider: {provider_info})")
        self.llm = LLMFactory.create_llm(
            config_path=self.llm_config_path,
            provider=self.llm_provider,
            model=self.model,
            temperature=self.temperature,
        )
        return self.llm

    def setup_database(self) -> Optional[DatabaseManager]:
        """
        Initialize the database manager (if SQL tools are enabled).

        Returns:
            DatabaseManager instance or None if SQL tools are disabled
        """
        if not self.enable_sql_tool:
            logger.info("Database manager disabled (SQL tools disabled)")
            self.db_manager = None
            self.custom_schema = {}
            return None

        # Get database URI from env if not provided
        db_uri = self.db_uri
        if db_uri is None:
            db_uri = os.getenv(
                "DATABASE_URI",
                "mssql+pyodbc://username:password@server/database"
                "?driver=ODBC+Driver+17+for+SQL+Server",
            )
            logger.debug("Database URI loaded from environment")

        logger.info("Setting up database manager...")
        self.db_manager = DatabaseManager(
            db_uri=db_uri,
            schema_path=self.schema_path,
            sample_rows=self.sample_rows,
            pool_size=self.pool_size,
            max_overflow=self.max_overflow,
            pool_recycle=self.pool_recycle,
        )

        # Custom schema is already loaded by DatabaseManager
        self.custom_schema = self.db_manager.custom_schema
        logger.debug(f"Schema loaded with {len(self.custom_schema)} tables")

        return self.db_manager

    def setup_tools(self) -> ToolsManager:
        """
        Initialize the tools manager.

        Returns:
            ToolsManager instance
        """
        if self.llm is None:
            raise RuntimeError(
                "LLM must be initialized before setting up tools. Call setup_llm() first."
            )

        logger.info("Setting up tools manager...")
        self.tools_manager = ToolsManager(
            db=self.db_manager.get_database() if self.db_manager else None,
            llm=self.llm,
            output_limit=self.output_limit,
            enable_sql_tool=self.enable_sql_tool,
        )
        return self.tools_manager

    def setup_prompt(self) -> Any:
        """
        Build the prompt template.

        Returns:
            Prompt template instance
        """
        logger.info("Building prompt template...")
        self.prompt_builder = PromptBuilder(system_prompt_path=self.system_prompt_path)
        self.prompt = self.prompt_builder.build_prompt(self.custom_schema)
        return self.prompt

    def setup_agent_factory(self) -> AgentFactory:
        """
        Create the agent factory.

        Returns:
            AgentFactory instance
        """
        if self.llm is None:
            raise RuntimeError("LLM must be initialized. Call setup_llm() first.")
        if self.tools_manager is None:
            raise RuntimeError("Tools manager must be initialized. Call setup_tools() first.")
        if self.prompt is None:
            raise RuntimeError("Prompt must be initialized. Call setup_prompt() first.")

        logger.info("Creating agent factory...")
        self.agent_factory = AgentFactory(
            llm=self.llm,
            tools=self.tools_manager.get_tools(),
            prompt=self.prompt,
            verbose=self.verbose,
            max_iterations=self.max_iterations,
            max_execution_time=self.max_execution_time,
        )
        return self.agent_factory

    def build_agent(self) -> Any:
        """
        Build the complete agent executor by initializing all components in order.

        Returns:
            Agent executor instance
        """
        logger.info("Building agent...")
        logger.debug(
            f"Model: {self.model}, Temperature: {self.temperature}, "
            f"SQL enabled: {self.enable_sql_tool}"
        )

        # Initialize components in correct order
        self.setup_llm()
        self.setup_database()
        self.setup_tools()
        self.setup_prompt()
        self.setup_agent_factory()

        # Create and return executor
        if self.agent_factory is None:
            raise RuntimeError("Agent factory creation failed")

        agent_executor = self.agent_factory.create_executor()
        logger.info("✅ Agent built successfully")
        return agent_executor
