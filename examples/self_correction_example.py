"""Example demonstrating Self-Correction and Reflection loops in LangGraph.

This script shows how to:
1. Define a state graph that validates LLM output.
2. Route back to the agent with error feedback if validation fails.
3. Terminate after successful validation or reaching a max retry limit.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import TypedDict, Annotated, cast
from collections.abc import Sequence
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from services.llm_factory import LLMFactory
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.message import add_messages  # pyright: ignore[reportMissingTypeStubs]
from langgraph.checkpoint.memory import MemorySaver

_ = load_dotenv()


# 1. Define the Graph State
class ReflectionState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    retry_count: int
    validation_error: str | None


async def main():
    print("🤖 Initializing Self-Correction & Reflection Example...")
    try:
        # Create LLM
        llm = LLMFactory.create_llm()

        # 2. Define the node functions
        def generator_node(state: ReflectionState):
            """Generates the requested JSON payload."""
            messages = list(state["messages"])
            retry_count = state["retry_count"]
            validation_error = state["validation_error"]

            # If there was a validation error, append it as feedback
            if validation_error:
                feedback = (
                    f"CRITICAL: The previous output failed validation with the following error:\n"
                    f"'{validation_error}'\n"
                    f"Please reflect on the error, correct your mistake, and generate the exact JSON format requested."
                )
                messages.append(HumanMessage(content=feedback))
                print(f"🔄 Generator Node: Retrying (Attempt {retry_count + 1})...")
            else:
                print("📝 Generator Node: Generating initial payload...")

            response = llm.invoke(messages)
            return {"messages": [response], "retry_count": retry_count + 1}

        def validator_node(state: ReflectionState):
            """Validates the generated output."""
            print("🔍 Validator Node: Checking output format...")
            last_message = state["messages"][-1]
            content = str(last_message.content).strip()

            # Try to extract JSON
            try:
                # Basic cleanup in case LLM wrapped it in markdown code blocks
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0].strip()

                parsed_data = cast(object, json.loads(content))
                if not isinstance(parsed_data, dict):
                    error_msg = "Expected JSON object."
                    print(f"❌ Validation Failed: {error_msg}")
                    return {"validation_error": error_msg}
                data = cast(dict[str, object], parsed_data)

                # Check for mandatory keys: name, email, role, and a specific check: verified must be True
                required_keys = ["name", "email", "role", "verified"]
                missing_keys = [k for k in required_keys if k not in data]

                if missing_keys:
                    error_msg = f"Missing mandatory keys: {', '.join(missing_keys)}"
                    print(f"❌ Validation Failed: {error_msg}")
                    return {"validation_error": error_msg}

                if not data.get("verified"):
                    error_msg = "The 'verified' boolean key must be set to true."
                    print(f"❌ Validation Failed: {error_msg}")
                    return {"validation_error": error_msg}

                # Validation passed! Clear errors
                print("✅ Validation Succeeded!")
                return {"validation_error": None}

            except json.JSONDecodeError as je:
                error_msg = f"Invalid JSON format. Error: {str(je)}"
                print(f"❌ Validation Failed: {error_msg}")
                return {"validation_error": error_msg}

        # Define the conditional routing logic
        def route_after_validation(state: ReflectionState):
            validation_error = state["validation_error"]
            retry_count = state["retry_count"]

            # If no errors, we are done
            if validation_error is None:
                return END

            # If we reached our maximum retry limit, force terminate
            if retry_count >= 3:
                print("⚠️ Max retries reached. Forcing completion.")
                return END

            # Otherwise, loop back to regenerate
            return "generator"

        # 3. Construct the LangGraph StateGraph
        workflow = StateGraph(ReflectionState)

        _ = workflow.add_node("generator", generator_node)  # pyright: ignore[reportUnknownMemberType]
        _ = workflow.add_node("validator", validator_node)  # pyright: ignore[reportUnknownMemberType]

        _ = workflow.add_edge(START, "generator")
        _ = workflow.add_edge("generator", "validator")
        _ = workflow.add_conditional_edges(
            "validator", route_after_validation, ["generator", END]
        )

        # Compile graph
        memory = MemorySaver()
        graph = workflow.compile(checkpointer=memory)  # pyright: ignore[reportUnknownMemberType]

        # 4. Run the experiment
        # We instruct the model to create a JSON but we intentionally make it tricky (like asking for a verified flag but not explicitly saying how)
        # to test the correction loop.
        thread_config: RunnableConfig = {
            "configurable": {"thread_id": "reflection_session_1"}
        }
        prompt = (
            "Generate a JSON object containing: name: 'Alice', email: 'alice@example.com', role: 'developer'. "
            "IMPORTANT: Do not output any markdown formatting, only the JSON block. Also make sure to set the 'verified' flag."
        )

        print(f"\nPrompt: {prompt}\n")

        result = cast(
            ReflectionState,
            await graph.ainvoke(  # pyright: ignore[reportUnknownMemberType]
                {
                    "messages": [HumanMessage(content=prompt)],
                    "retry_count": 0,
                    "validation_error": None,
                },
                config=thread_config,
            ),
        )

        # Print final result
        print("\n🏁 Final Output from Agent:")
        print("-" * 50)
        print(result["messages"][-1].content)
        print("-" * 50)
        print(f"Total attempts: {result['retry_count']}")
        print(
            f"Final Validation Status: {'Failed' if result['validation_error'] else 'Passed'}"
        )

    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
