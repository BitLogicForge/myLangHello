from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

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


class AgentApp:
    def __init__(self):
        load_dotenv()
        self.llm = ChatOpenAI(temperature=0, model="gpt-4")
        self.db = self._init_db()
        self.custom_table_info = self._load_custom_schema()
        self.tools = self._register_tools()
        self.prompt = self._create_prompt()
        self.agent = self._create_agent()
        self.agent_executor = self._create_executor()

    def _init_db(self):
        return SQLDatabase.from_uri(
            #   "mssql+pyodbc://user:pass@host/dbname?driver=ODBC+Driver+17+for+SQL+Server",
            "sqlite:///example.db",
            include_tables=["users", "orders", "products"],
            sample_rows_in_table_info=2,
        )

    def _load_custom_schema(self):
        import json

        with open("db_schema_config.json", "r", encoding="utf-8") as f:
            return json.load(f)

    def _register_tools(self):
        sql_toolkit = SQLDatabaseToolkit(db=self.db, llm=self.llm)
        sql_tools = sql_toolkit.get_tools()
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

    def _create_prompt(self):
        system_prompt = self.load_txt("messages/system_prompt.txt")

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", "{input}"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        schema_context = "\n".join(
            [f"- {table}: {desc}" for table, desc in self.custom_table_info.items()]
        )
        return prompt.partial(schema_context=schema_context)

    def _create_agent(self):
        return create_openai_functions_agent(self.llm, self.tools, self.prompt)

    def _create_executor(self):
        return AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,
        )

    def load_txt(self, path="messages/system_prompt.txt"):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def run(self, question=None):
        if question is None:
            question = (
                "How many users are registered in the database? "
                "What are the total sales from completed orders? "
                "Also, what's 5 + 7 and what's the weather in Paris?"
            )
        response = self.agent_executor.invoke({"input": question})
        print("\n" + "=" * 50)
        print("Final Agent Response:")
        print("=" * 50)
        print(response["output"])


def main():
    print("Hello, Function Calling Agent!")
    app = AgentApp()
    app.run()


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
