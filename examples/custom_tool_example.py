"""Example demonstrating how to write custom tools with structured input validation.

This script shows how to:
1. Define a Pydantic schema for tool arguments.
2. Create a custom tool using the LangChain `@tool` decorator.
3. Initialize a quick agent that includes this custom tool.
4. Execute queries where the agent utilizes the tool.
"""

import asyncio
import sys
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from services.llm_factory import LLMFactory
from langchain.agents import create_agent

load_dotenv()


# 1. Define the input schema for your tool
class MealPlannerInput(BaseModel):
    """Input parameters for the meal planner tool."""
    main_ingredient: str = Field(
        ..., 
        description="The primary ingredient for the meal (e.g. 'chicken', 'tofu', 'salmon')"
    )
    vegetables: list[str] = Field(
        default=[],
        description="A list of vegetables available to include in the recipe"
    )
    max_prep_time_minutes: int = Field(
        default=30,
        description="Maximum preparation time allowed in minutes"
    )
    dietary_restriction: str | None = Field(
        default=None,
        description="Optional dietary restriction (e.g., 'vegetarian', 'gluten-free', 'vegan')"
    )


# 2. Create the tool using the @tool decorator and the schema
@tool(args_schema=MealPlannerInput)
def suggest_recipe_tool(
    main_ingredient: str,
    vegetables: list[str],
    max_prep_time_minutes: int = 30,
    dietary_restriction: str | None = None
) -> str:
    """Suggests a recipe based on main ingredient, available vegetables, time limits, and dietary options."""
    
    # Custom business logic / database lookup / external API call
    veg_str = ", ".join(vegetables) if vegetables else "none"
    restriction_str = f" ({dietary_restriction})" if dietary_restriction else ""
    
    # Simulated recipe lookup logic
    recipe_name = f"Quick {main_ingredient.capitalize()} & Vegetable Stir-Fry"
    if dietary_restriction == "vegetarian" or main_ingredient.lower() in ["tofu", "beans"]:
        recipe_name = f"Garden Fresh {main_ingredient.capitalize()} Medley"
    
    return (
        f"--- Recipe Suggestion --- \n"
        f"Recipe: {recipe_name}{restriction_str}\n"
        f"Main Ingredient: {main_ingredient}\n"
        f"Vegetables used: {veg_str}\n"
        f"Prep Time: {max_prep_time_minutes} minutes\n"
        f"Instructions: Chop everything, toss in a hot pan with olive oil, seasoning, and cook until tender."
    )


async def main():
    print("🤖 Initializing LLM and building custom agent...")
    try:
        # Create LLM
        llm = LLMFactory.create_llm()
        
        # Build list of tools including our new custom tool
        tools = [suggest_recipe_tool]
        
        # Create a simple agent with our tool
        # (We use the project's base prompt style or a standard react style prompt)
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=(
                "You are a helpful culinary assistant. You have access to a tool "
                "called 'suggest_recipe_tool' which takes structured ingredients and restriction details. "
                "Always use this tool when someone asks for recipe ideas based on what they have."
            )
        )
        
        # Question that forces the agent to extract parameters from user input
        question = (
            "I want a vegetarian dinner recipe. I have tofu, carrots, and spinach. "
            "I only have 20 minutes to cook. What should I make?"
        )
        
        print(f"\n💬 User Question: {question}")
        print("⏳ Invoking Agent...")
        
        # Invoke agent
        response = await agent.ainvoke({"messages": [HumanMessage(content=question)]})  # type: ignore
        
        print("\n✅ Agent Output:")
        print("-" * 50)
        messages_list = response.get("messages", [])
        if messages_list:
            # Print the final assistant message
            print(messages_list[-1].content)
        else:
            print(str(response))
        print("-" * 50)
        
    except Exception as e:
        print(f"\n❌ Error building or running custom tool agent: {e}")


if __name__ == "__main__":
    asyncio.run(main())
