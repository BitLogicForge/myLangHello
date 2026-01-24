from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
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


# Define tools with structured schemas for function calling
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
    print("Hello, Function Calling Agent!")

    # Use gpt-4 or gpt-3.5-turbo-1106+ for function calling support
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Initialize database
    import json

    db = SQLDatabase.from_uri(
        "sqlite:///example.db",
        include_tables=["users", "orders", "products"],
        sample_rows_in_table_info=2,
    )

    # Load custom schema info
    with open("db_schema_config.json", "r", encoding="utf-8") as f:
        custom_table_info = json.load(f)

    # Create SQL toolkit
    sql_toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = sql_toolkit.get_tools()

    # Register all tools
    tools = [
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

    # Create a prompt template for function calling agent
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant with access to various tools including a database.

Database Schema Context:
{schema_context}

When querying the database:
- Always check the schema first if unsure about table structure
- Use proper SQL syntax for SQLite
- Be careful with date formats and data types
- Status values in orders table are: 'pending', 'completed', 'cancelled'
- Categories in products table are: 'electronics', 'clothing', 'food'
- Always use JOIN when combining data from multiple tables
- Use aggregate functions (COUNT, SUM, AVG) for statistical queries

Always think step-by-step and use the appropriate tools to answer questions accurately.""",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Format schema context
    schema_context = "\n".join([f"- {table}: {desc}" for table, desc in custom_table_info.items()])

    # Partial the prompt with schema context
    prompt = prompt.partial(schema_context=schema_context)

    # Create the OpenAI Functions agent
    agent = create_openai_functions_agent(llm, tools, prompt)

    # Create agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
    )

    # Example questions
    question = (
        "How many users are registered in the database? "
        "What are the total sales from completed orders? "
        "Also, what's 5 + 7 and what's the weather in Paris?"
    )

    # Run the agent
    response = agent_executor.invoke({"input": question})

    print("\n" + "=" * 50)
    print("Final Agent Response:")
    print("=" * 50)
    print(response["output"])


if __name__ == "__main__":
    main()
