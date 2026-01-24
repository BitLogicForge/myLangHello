from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
# example with manual logic to use tools in an agent-like manner


# Example tool 1: Placeholder for a calculator tool
def calculator_tool(expression: str) -> str:
    """Evaluates a simple math expression and returns the result as a string."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# Example tool 2: Placeholder for a weather lookup tool
def weather_tool(city: str) -> str:
    """Returns a fake weather report for the given city."""
    # Placeholder logic
    return f"The weather in {city} is sunny and 25°C."


def main():
    print("Hello, agent with tools!")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # Simulate agent receiving a question and using tools
    question = "What is 5 + 7? Also, what's the weather in Paris?"

    # Agent logic: detect which tool to use (placeholder logic)
    if "+" in question or any(char.isdigit() for char in question):
        calc_result = calculator_tool("5 + 7")
        print(f"Calculator Tool Used: {calc_result}")
    else:
        calc_result = None

    if "weather" in question.lower():
        weather_result = weather_tool("Paris")
        print(f"Weather Tool Used: {weather_result}")
    else:
        weather_result = None

    # Build conversation history for LLM
    conversation_history = [
        SystemMessage(content="You are an assistant that can use tools to answer questions."),
        HumanMessage(content=question),
    ]
    if calc_result:
        conversation_history.append(AIMessage(content=calc_result))
    if weather_result:
        conversation_history.append(AIMessage(content=weather_result))

    # Ask LLM to summarize the answer using the tool results
    conversation_history.append(HumanMessage(content="Summarize the answer for me."))
    response = llm.invoke(conversation_history)

    print("Response from LLM:")
    if hasattr(response, "content"):
        print(response.content)
    else:
        print(response)


if __name__ == "__main__":
    main()
