from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

from base_chain_tools import (
    calculator_tool,
    weather_tool,
    read_file_tool,
    write_file_tool,
    http_get_tool,
    list_dir_tool,
    system_info_tool,
    random_joke_tool,
)

load_dotenv()


# Small wrapper functions exposed as LangChain tools. They keep simple string
# signatures so agents can call them easily.
@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression and return the result as a string."""
    return calculator_tool(expression)


@tool
def weather(city: str) -> str:
    """Return a fake weather report for the given city."""
    return weather_tool(city)


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents or an error message."""
    return read_file_tool(path)


@tool
def write_file(spec: str) -> str:
    """Write content to a file. Spec: first line is path, rest is content."""
    return write_file_tool(spec)


@tool
def http_get(url: str) -> str:
    """Perform an HTTP GET and return a short summary/result."""
    return http_get_tool(url)


@tool
def list_dir(path: str = ".") -> str:
    """List files in a directory and return a newline-separated listing."""
    return list_dir_tool(path)


@tool
def system_info(query: str = "") -> str:
    """Return basic system information. The query parameter is ignored."""
    return system_info_tool()


@tool
def random_joke(query: str = "") -> str:
    """Return a small, harmless random joke. The query parameter is ignored."""
    return random_joke_tool()


def main():
    print("Hello, SQL-enabled agent!")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # Initialize database with custom descriptions
    # Example: Replace with your actual database URI
    # PostgreSQL: "postgresql://user:password@localhost/dbname"
    # MySQL: "mysql+pymysql://user:password@localhost/dbname"
    db = SQLDatabase.from_uri(
        "sqlite:///example.db",  # Replace with your database
        include_tables=["users", "orders", "products"],  # Limit to specific tables
        sample_rows_in_table_info=2,  # Include sample data in schema
    )

    # Load custom context about your database schema from JSON file
    import json

    with open("db_schema_config.json", "r", encoding="utf-8") as f:
        custom_table_info = json.load(f)

    # Create SQL toolkit - provides multiple SQL-related tools
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    # Get all SQL tools from the toolkit
    # This includes: sql_db_query, sql_db_schema, sql_db_list_tables, sql_db_query_checker
    sql_tools = sql_toolkit.get_tools()

    # Register all tools (custom tools + SQL tools)
    tools = [
        calculator,
        weather,
        read_file,
        write_file,
        http_get,
        list_dir,
        system_info,
        random_joke,
        *sql_tools,  # Unpack SQL tools here
    ]

    # Create agent with custom system message that includes database context
    system_message = f"""You are a helpful assistant with access to various tools including a database.

Database Schema Context:
{chr(10).join([f"- {table}: {desc}" for table, desc in custom_table_info.items()])}

When querying the database:
- Always check the schema first if unsure about table structure
- Use proper SQL syntax for SQLite (or your database type)
- Be careful with date formats and data types
- Status values in orders table are: 'pending', 'completed', 'cancelled'
- Categories in products table are: 'electronics', 'clothing', 'food'
- Always use JOIN when combining data from multiple tables
- Use aggregate functions (COUNT, SUM, AVG) for statistical queries

For questions about users, orders, or products, use the database tools.
For calculations, use the calculator tool.
For weather, use the weather tool.
"""

    # Initialize agent with the custom system message
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        agent_kwargs={"prefix": system_message},
    )

    # Example questions that combine different tools
    question = (
        "How many users are registered in the database? "
        "What are the total sales from completed orders? "
        "Also, what's 5 + 7 and what's the weather in Paris?"
    )

    # The agent will decide which tool(s) to use automatically
    response = agent.run(question)

    print("\nFinal Agent Response:")
    print(response)


if __name__ == "__main__":
    main()
