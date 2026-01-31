from langchain_core.messages import AIMessage, HumanMessage


def prepare_messages_with_history(
    question: str, history: list[tuple[str, str]] | None = None
) -> list[HumanMessage | AIMessage]:
    """
    Prepare messages for LangGraph agent with optional conversation history.

    Args:
        question: Current user question
        history: Optional conversation history as list of (role, content) tuples
                Example: [("user", "Hello"), ("assistant", "Hi!"), ...]

    Returns:
        List of properly formatted LangChain messages
    """
    messages: list[HumanMessage | AIMessage] = []

    # Add history if provided
    if history:
        for role, content in history:
            if role.lower() in ["user", "human"]:
                messages.append(HumanMessage(content=content))
            elif role.lower() in ["assistant", "ai"]:
                messages.append(AIMessage(content=content))

    # Append current question
    messages.append(HumanMessage(content=question))

    return messages
