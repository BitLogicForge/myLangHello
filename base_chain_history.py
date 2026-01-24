from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()


def main():
    print("Hello, manual conversation history!")
    llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")

    # Example manual conversation history
    conversation_history = [
        SystemMessage(content="You are a helpful assistant that suggests creative company names."),
        HumanMessage(content="What is a good name for a company that makes colorful socks?"),
        AIMessage(content="Rainbow Threads"),
        HumanMessage(content="any other suggestions?"),
    ]

    response = llm.invoke(conversation_history)

    print("Response from LLM:")
    if hasattr(response, "content"):
        print(response.content)
    else:
        print(response)


if __name__ == "__main__":
    main()
