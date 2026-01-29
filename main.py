"""Refactored Agent Application with Single Responsibility Principle."""

import os
import logging
from typing import Optional
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from services import DatabaseManager, ToolsManager, PromptBuilder, AgentFactory

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class AgentApp:
    """Main application orchestrator - coordinates all components."""

    def __init__(
        self,
        db_uri: Optional[str] = None,
        model: str = "gpt-4",
        temperature: float = 0.7,
        schema_path: str = "db_schema_config.json",
        enable_sql_tool: bool = True,
    ):
        """
        Initialize the agent application.

        Args:
            db_uri: Database connection URI (MSSQL format).
                   If None, reads from DATABASE_URI env var.
                   Example:
                   mssql+pyodbc://username:password@server/database
                   ?driver=ODBC+Driver+17+for+SQL+Server

                   Or with Windows Auth:
                   mssql+pyodbc://server/database
                   ?driver=ODBC+Driver+17+for+SQL+Server
                   &trusted_connection=yes
            model: LLM model name
            temperature: LLM temperature setting
            schema_path: Path to database schema configuration
            enable_sql_tool: Enable or disable SQL query tool
        """
        logger.info("Initializing AgentApp...")
        logger.debug(f"Model: {model}, Temperature: {temperature}, SQL enabled: {enable_sql_tool}")

        # Get database URI from env if not provided
        if db_uri is None:
            db_uri = os.getenv(
                "DATABASE_URI",
                "mssql+pyodbc://username:password@server/database"
                "?driver=ODBC+Driver+17+for+SQL+Server",
            )
            logger.debug("Database URI loaded from environment")

        # Initialize core components
        # ChatOpenAI Parameters (categorized by importance):

        # *** ESSENTIAL PARAMETERS ***
        # model: str - Model name (e.g., "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo")
        # temperature: float (0-2) - Creativity level. 0=deterministic, 2=very creative

        # *** IMPORTANT PARAMETERS ***
        # max_tokens: int - Maximum tokens in response (None=unlimited by model)
        # timeout: float - Request timeout in seconds (default: 60)
        # max_retries: int - Number of retry attempts on failure (default: 2)
        # api_key: str - OpenAI API key (defaults to OPENAI_API_KEY env var)

        # *** CONTROL & TUNING ***
        # streaming: bool - Enable token-by-token streaming (default: False)
        # n: int - Number of completions to generate (default: 1)
        # top_p: float (0-1) - Nucleus sampling (alternative to temperature)
        # frequency_penalty: float (-2 to 2) - Reduce repetition of frequent tokens
        # presence_penalty: float (-2 to 2) - Encourage new topics
        # logit_bias: dict - Modify likelihood of specific tokens

        # *** FUNCTION CALLING & STRUCTURED OUTPUT ***
        # model_kwargs: dict - Additional args like response_format, seed, etc.
        # function_call: Optional[dict] - Control function calling behavior

        # *** ADVANCED & ORGANIZATIONAL ***
        # request_timeout: float - Alternative to timeout parameter
        # base_url: str - Custom API endpoint (for proxies/testing)
        # organization: str - OpenAI organization ID
        # openai_proxy: str - HTTP proxy for requests
        # tiktoken_model_name: str - Override tokenizer model
        # default_headers: dict - Custom HTTP headers
        # default_query: dict - Additional query parameters
        # http_client: Any - Custom HTTP client
        # verbose: bool - Enable detailed logging (default: False)

        # *** CACHING & CALLBACKS ***
        # cache: bool - Enable prompt caching (default: True)
        # callbacks: List - Callback handlers for monitoring
        # tags: List[str] - Tags for tracking/filtering
        # metadata: dict - Additional metadata for requests

        logger.info(f"Initializing LLM: {model}")
        self.llm = ChatOpenAI(
            temperature=temperature,
            model=model,
            # Uncomment and configure as needed:
            # max_tokens=4096,
            # timeout=120,
            # streaming=True,
            # max_retries=3,
            # verbose=True,
        )

        # Setup database
        logger.info("Setting up database manager...")
        self.db_manager = DatabaseManager(
            db_uri=db_uri,
            include_tables=["users", "orders", "products"],
            sample_rows=2,
            # Connection pool configuration for ~100 users
            # Adjust based on your specific load pattern
            pool_size=10,  # Increased from default 5
            max_overflow=15,  # Increased from default 10
            pool_recycle=1800,  # 30 minutes (reduced from 3600)
        )

        # Load custom schema
        logger.info(f"Loading custom schema from {schema_path}")
        self.custom_schema = DatabaseManager.load_custom_schema(schema_path)
        logger.debug(f"Schema loaded with {len(self.custom_schema)} tables")

        # Setup tools
        logger.info("Setting up tools manager...")
        self.tools_manager = ToolsManager(
            db=self.db_manager.get_database(),
            llm=self.llm,
            output_limit=10000,
            enable_sql_tool=enable_sql_tool,
        )

        # Build prompt
        logger.info("Building prompt template...")
        self.prompt_builder = PromptBuilder(system_prompt_path="messages/system_prompt.txt")
        self.prompt = self.prompt_builder.build_prompt(self.custom_schema)

        # Create agent executor
        logger.info("Creating agent executor...")
        self.agent_factory = AgentFactory(
            llm=self.llm,
            tools=self.tools_manager.get_tools(),
            prompt=self.prompt,
            verbose=True,
            max_iterations=10,
            max_execution_time=60,
        )
        self.agent_executor = self.agent_factory.create_executor()
        logger.info("✅ AgentApp initialized successfully")

    def run(self, question: str) -> dict:
        """
        Run the agent with a question.

        Args:
            question: User question/input

        Returns:
            Agent response dictionary
        """
        logger.info("Running agent with question...")
        logger.debug(f"Question: {question[:100]}...")

        try:
            # Standard AgentExecutor input format
            response = self.agent_executor.invoke({"input": question})
            logger.info("✅ Agent completed successfully")
            return response
        except Exception as e:
            logger.error(f"❌ Agent execution failed: {str(e)}")
            raise

    @staticmethod
    def _print_response(response: dict) -> None:
        """Print formatted agent response."""
        print("\n" + "=" * 50)
        print("Final Agent Response:")
        print("=" * 50)
        print(response["output"])


def main():
    """Main entry point."""
    print("Hello, Function Calling Agent!")
    question = (
        # "How many users are registered in the database? "
        # "What are the total sales from completed orders? "
        "Also, what's 5 + 7 and what's the weather in Paris?"
    )
    app = AgentApp(enable_sql_tool=False)  # Set to False to disable SQL tool
    app.run(question=question)


if __name__ == "__main__":
    main()
