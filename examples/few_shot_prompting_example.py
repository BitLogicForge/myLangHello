"""Example demonstrating Dynamic Few-Shot Prompting with Pydantic structured output.

This script shows how to:
1. Define a Pydantic schema for structured output.
2. Maintain a collection of few-shot prompt examples.
3. Dynamically inject these examples into a system prompt.
4. Execute and validate the structured response from the LLM.
"""

import asyncio
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, cast
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from services.llm_factory import LLMFactory

load_dotenv()


# 1. Define the Pydantic Schema for target database filters
class QueryFilter(BaseModel):
    column: str = Field(..., description="The name of the database column to filter on")
    operator: str = Field(..., description="Comparison operator (e.g. '=', '>', '<', 'LIKE')")
    value: str = Field(..., description="The value to compare against")


class DatabaseQueryIntent(BaseModel):
    """Structured representation of the database query intent."""
    table: str = Field(..., description="The table name to query (e.g. 'users', 'orders', 'products')")
    filters: List[QueryFilter] = Field(..., description="List of columns, operators, and values to filter on")
    limit: Optional[int] = Field(None, description="Optional limit of rows to return")


# 2. Static few-shot examples (representing NLP-to-SQL/Filter conversions)
FEW_SHOT_EXAMPLES = [
    {
        "input": "find all active users from London",
        "output": {
            "table": "users",
            "filters": [
                {"column": "status", "operator": "=", "value": "active"},
                {"column": "city", "operator": "=", "value": "London"}
            ]
        }
    },
    {
        "input": "get the top 5 products cheaper than 20 dollars",
        "output": {
            "table": "products",
            "filters": [
                {"column": "price", "operator": "<", "value": "20"}
            ],
            "limit": 5
        }
    },
    {
        "input": "show me orders pending delivery shipped after May 1st",
        "output": {
            "table": "orders",
            "filters": [
                {"column": "status", "operator": "=", "value": "pending"},
                {"column": "ship_date", "operator": ">", "value": "2026-05-01"}
            ]
        }
    }
]


# 3. Dynamic Prompt Builder helper
def build_few_shot_prompt(user_query: str) -> str:
    """Constructs the system prompt with few-shot examples injected."""
    prompt_header = (
        "You are an NLP-to-Database translation agent. Your task is to convert the user's natural language "
        "query into a structured query intent JSON matching the requested schema. "
        "Study the examples below to understand the expected table names, columns, operators, and formatting:\n\n"
    )
    
    examples_str = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES):
        examples_str += f"### Example {i+1}\n"
        examples_str += f"Input: {ex['input']}\n"
        # Format output as compact string for the prompt
        examples_str += f"Output Structure:\n{ex['output']}\n\n"
        
    prompt_footer = (
        f"### New Request\n"
        f"Input: {user_query}\n"
        f"Output Structure:\n"
    )
    
    return prompt_header + examples_str + prompt_footer


async def main():
    print("🤖 Initializing Dynamic Few-Shot Prompting Example...")
    try:
        # Create LLM
        llm = LLMFactory.create_llm()
        
        # 4. Bind the structured output schema to the LLM
        # This guarantees Pylance/mypy compatibility and forces the model output type
        structured_llm = llm.with_structured_output(DatabaseQueryIntent)
        
        # 5. Define test queries
        test_queries = [
            "find products with stock level greater than 100",
            "get 10 customers registered after June 15th with status vip"
        ]
        
        for query in test_queries:
            print(f"\n💬 Natural Language: '{query}'")
            
            # Format the prompt dynamically with few-shot examples
            prompt = build_few_shot_prompt(query)
            
            print("⏳ Translating using structured model + few-shot examples...")
            result = cast(DatabaseQueryIntent, await structured_llm.ainvoke(prompt))
            
            # Print parsed Pydantic output
            print("✅ Parsed Database Intent:")
            print(f"   Table: {result.table}")
            if result.limit:
                print(f"   Limit: {result.limit}")
            print("   Filters:")
            for f in result.filters:
                print(f"     - {f.column} {f.operator} '{f.value}'")
            print("-" * 50)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Note: Ensure your configured provider supports structured output (e.g. OpenAI or newer Ollama models).")


if __name__ == "__main__":
    asyncio.run(main())
