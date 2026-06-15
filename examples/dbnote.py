from langchain_community.agent_toolkits.sql.base import create_sql_agent  # pyright: ignore[reportUnknownVariableType]
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

toolkit_kwargs: dict[str, object] = {
    "db": SQLDatabase.from_uri(  # pyright: ignore[reportUnknownMemberType]
        database_uri="postgresql+psycopg2://root:password@localhost:5432/dev_db"
    ),
    "llm": llm,
}

agent = create_sql_agent(
    llm=llm,
    agent_type="openai-tools",
    verbose=True,
    toolkit=SQLDatabaseToolkit(**toolkit_kwargs),  # pyright: ignore[reportArgumentType]
)
