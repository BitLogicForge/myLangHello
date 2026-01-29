"""Example: Switching between Azure OpenAI and OpenAI."""

from dotenv import load_dotenv
from main import AgentApp
from services import LLMFactory

# Load environment variables
load_dotenv()


def example_1_using_config_file():
    """Example 1: Use the provider configured in llm_config.json."""
    print("\n=== Example 1: Using Config File ===")

    # Simply create app - it reads llm_config.json automatically
    app = AgentApp()

    result = app.run("What's 25 * 4?")
    print(f"Result: {result}")


def example_2_override_provider():
    """Example 2: Override provider at runtime."""
    print("\n=== Example 2: Override Provider ===")

    # Force using OpenAI even if config says Azure
    app = AgentApp(llm_provider="openai")

    result = app.run("What's the weather like?")
    print(f"Result: {result}")


def example_3_override_parameters():
    """Example 3: Override specific parameters."""
    print("\n=== Example 3: Override Parameters ===")

    # Override multiple parameters
    app = AgentApp(
        llm_provider="azure",
        model="gpt-4",
        temperature=0.9,  # More creative
    )

    result = app.run("Tell me a random joke")
    print(f"Result: {result}")


def example_4_direct_llm_factory():
    """Example 4: Use LLMFactory directly for custom scenarios."""
    print("\n=== Example 4: Direct LLM Factory ===")

    # Create Azure LLM
    azure_llm = LLMFactory.create_llm(provider="azure", model="gpt-4", temperature=0.7)
    print(f"Created Azure LLM: {type(azure_llm).__name__}")

    # Create OpenAI LLM
    openai_llm = LLMFactory.create_llm(provider="openai", model="gpt-3.5-turbo", temperature=0.5)
    print(f"Created OpenAI LLM: {type(openai_llm).__name__}")

    # Use directly with LangChain
    response = azure_llm.invoke("Say hello!")
    print(f"Response: {response.content}")


def example_5_switching_providers():
    """Example 5: Dynamically switch between providers."""
    print("\n=== Example 5: Dynamic Provider Switching ===")

    providers = ["azure", "openai"]
    question = "What is 2+2?"

    for provider in providers:
        print(f"\n--- Testing with {provider.upper()} ---")
        try:
            app = AgentApp(llm_provider=provider)
            app.run(question)
            print(f"Success with {provider}")
        except Exception as e:
            print(f"Error with {provider}: {e}")


if __name__ == "__main__":
    print("LLM Configuration Examples")
    print("=" * 60)

    # Check available providers
    providers = LLMFactory.get_available_providers()
    print(f"\nAvailable providers: {providers}")

    # Run examples (uncomment the ones you want to test)

    # example_1_using_config_file()
    # example_2_override_provider()
    # example_3_override_parameters()
    # example_4_direct_llm_factory()
    # example_5_switching_providers()

    print("\n" + "=" * 60)
    print("Examples complete!")
