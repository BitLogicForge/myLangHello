from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType

load_dotenv()


# Define tools using LangChain's @tool decorator
@tool
def calculator(expression: str) -> str:
    """Evaluates a simple math expression and returns the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


@tool
def weather(city: str) -> str:
    """Returns a fake weather report for the given city."""
    return f"The weather in {city} is sunny and 25°C."


def main():
    print("Hello, state-of-the-art agent!")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # Register tools
    tools = [calculator, weather]

    # Initialize a zero-shot agent that can use tools
    agent = initialize_agent(
        tools,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    # User question that requires tool use
    question = "What is 5 + 7? Also, what's the weather in Paris?"

    # The agent will decide which tool(s) to use automatically
    response = agent.run(question)

    print("\nFinal Agent Response:")
    print(response)


if __name__ == "__main__":
    main()
