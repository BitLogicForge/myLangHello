from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI

agent = create_sql_agent(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    agent_type="openai-tools",
    verbose=True,
    toolkit=SQLDatabaseToolkit(
        db=SQLDatabase.from_uri(
            database_uri="postgresql+psycopg2://root:password@localhost:5432/dev_db"
        ),
        llm=ChatOpenAI(model="gpt-4o-mini"),
    ),
)

agent.invoke({"input": "How many calls happenend?"})
