from langchain_core.messages import AIMessage, HumanMessage
from typing import TypeGuard


# MARK: Message Prep
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


# MARK: Type Guards
def is_str_dict(val: object) -> TypeGuard[dict[str, object]]:
    """Type guard to check if a value is a dictionary with string keys."""
    return isinstance(val, dict)


def is_list(val: object) -> TypeGuard[list[object]]:
    """Type guard to check if a value is a list of objects."""
    return isinstance(val, list)


def is_tuple(val: object) -> TypeGuard[tuple[object, ...]]:
    """Type guard to check if a value is a tuple."""
    return isinstance(val, tuple)
