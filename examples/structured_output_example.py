"""Example demonstrating structured output from LLMs in LangChain.

This script shows how to enforce the LLM to return a validated Pydantic model
instead of a natural language string.
"""

import asyncio
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, cast

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from services.llm_factory import LLMFactory
from dotenv import load_dotenv

load_dotenv()


# 1. Define the Pydantic schema for the structured response we want
class RecipeAndBudgetAnalysis(BaseModel):
    """Structured response containing budget calculation and recipe recommendations."""
    can_afford_all: bool = Field(
        ..., 
        description="True if the total estimated cost of all desired groceries is within the budget"
    )
    total_estimated_cost: float = Field(
        ..., 
        description="The estimated total cost of the desired groceries"
    )
    remaining_budget: float = Field(
        ..., 
        description="The remaining money after buying the groceries (budget - estimated cost)"
    )
    affordable_items: List[str] = Field(
        ...,
        description="List of desired items that CAN be bought within the budget"
    )
    missing_items: List[str] = Field(
        ...,
        description="List of desired items that CANNOT be bought within the budget"
    )
    suggested_recipes: List[str] = Field(
        ..., 
        description="1-3 recipes we can cook using the vegetables in the basket and/or groceries"
    )
    explanation: str = Field(
        ..., 
        description="A short explanation of the cost estimates, budget check, and recipe selections"
    )


async def main():
    print("🤖 Initializing LLM...")
    try:
        # Create LLM using the project's configured provider
        llm = LLMFactory.create_llm()
        
        # 2. Bind the Pydantic model to the LLM
        print("🔗 Binding Pydantic schema to the model (enforcing structured output)...")
        structured_llm = llm.with_structured_output(RecipeAndBudgetAnalysis)
        
        # 3. Define the structured inputs (representing frontend inputs)
        basket = ["tomato", "bell pepper", "onion", "garlic"]
        budget = 25.0
        desire_to_buy = ["chicken breast", "olive oil", "pasta", "parmesan cheese", "avocado"]
        query = (
            "Can I buy all these groceries with my budget? "
            "Suggest what I can cook using my basket vegetables and these new groceries."
        )
        
        # Format the final prompt to pass to the model
        formatted_prompt = (
            f"Basket of vegetables: {', '.join(basket)}\n"
            f"Available budget: ${budget:.2f}\n"
            f"Groceries to buy: {', '.join(desire_to_buy)}\n"
            f"Question: {query}\n"
        )
        
        print("\n📝 Formatted Prompt sent to LLM:")
        print("-" * 50)
        print(formatted_prompt)
        print("-" * 50)
        
        print("⏳ Invoking LLM and waiting for structured response...")
        result = cast(RecipeAndBudgetAnalysis, await structured_llm.ainvoke(formatted_prompt))
        
        # 5. Output the structured results
        print("\n✅ Structured Response Received:")
        print("-" * 50)
        print(f"Can Afford All: {result.can_afford_all}")
        print(f"Total Cost:     ${result.total_estimated_cost:.2f}")
        print(f"Remaining:      ${result.remaining_budget:.2f}")
        print(f"Affordable:     {result.affordable_items}")
        print(f"Missing:        {result.missing_items}")
        print(f"Recipes:        {result.suggested_recipes}")
        print(f"Explanation:    {result.explanation}")
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("Note: Ensure your selected provider supports structured output (e.g. OpenAI gpt-3.5-turbo/gpt-4, or newer Ollama models).")


if __name__ == "__main__":
    asyncio.run(main())
