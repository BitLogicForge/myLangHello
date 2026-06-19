# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose & Guiding Principle

**This project is a demo/proof-of-concept for an AI-powered chatbot backend specializing in stock performance analysis.**

The primary purpose is to demonstrate core functionality that can be expanded into a production-ready web application for displaying and analyzing stock performance. When working on this project, align all decisions with this principle: **we are building a foundational backend for a stock-focused chatbot that can answer financial questions, analyze market data, and help users make informed investment decisions.**

### What This Means for Development

- **Stock Market Focus**: Prioritize tools, features, and examples that relate to stock data, financial analysis, and market information
- **Chatbot Backend**: Design APIs and responses with conversational interfaces in mind
- **Demo Quality**: Maintain clean, demonstrable code that showcases the concept rather than production-hardened engineering
- **Expandable Foundation**: Build patterns and structures that can scale to a full stock analysis platform

## Project Overview

This is a FastAPI + LangGraph demo application that creates a tool-using AI agent with SQL database access capabilities. The agent supports multiple LLM providers (OpenAI, Azure OpenAI, LM Studio, Ollama) and includes custom tools, streaming responses, and Prometheus telemetry.

## Development Commands

### Running the Application

```bash
# Run the FastAPI server (recommended)
uv run python api.py

# Run with standard Python
python api.py

# Run console test application
uv run python examples/main_example.py
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_tools.py

# Run with verbose output
uv run pytest -v

# Run specific test
uv run pytest tests/test_tools.py::test_calculator_tool
```

### Code Quality

```bash
# Linting
uv run ruff check .

# Format code
uv run ruff format .

# Type checking
uv run mypy .

# Run all quality checks
uv run ruff check . && uv run mypy .
```

### Dependency Management

```bash
# Update dependencies (uses uv)
uv sync

# Add new dependency
uv add package-name

# Add dev dependency
uv add --dev package-name
```

## Architecture

### Core Components

The application follows a layered architecture with clear separation of concerns:

**AgentApp (main.py)**: Main orchestrator that coordinates all components. Handles initialization and execution flow.

**AgentConfigurator (services/agent_configurator.py)**: Factory that builds the complete agent with all dependencies (LLM provider, tools, prompts).

**AgentFactory (services/agent_factory.py)**: Creates the LangGraph agent with configured tools and guardrails.

**AgentRunner (services/agent_runner.py)**: Executes the agent with runtime safeguards (timeout, tool-call limits, iteration limits).

**LLMFactory (services/llm_factory.py)**: Provider factory that selects and configures the appropriate LLM provider based on config.json.

### API Layer

- **api.py**: FastAPI application entry point with middleware and route registration
- **routes/agent_routes.py**: Main `/query` endpoint for agent interactions
- **routes/health_routes.py**: Health check endpoint
- **routes/config_routes.py**: Configuration inspection endpoint

### Tool System

- **tools.py**: Custom tool implementations (calculator, weather, file I/O, HTTP, etc.) - **add stock-specific tools here**
- **services/tools_manager.py**: Registers custom tools and integrates with LangChain SQL toolkit
- **Table Info**: table_info.py and table_info_parse.py manage database schema metadata

**Stock-Specific Tools to Consider Adding**:
- Real-time stock price queries
- Historical price data retrieval
- Financial metrics (P/E ratio, market cap, volume, etc.)
- Portfolio return calculations
- Risk metrics (volatility, beta, drawdown analysis)
- Technical indicators (moving averages, RSI, MACD)
- Company financial statements and ratios
- Market index comparisons
- Currency conversion for international stocks

### Configuration

- **config.py**: Singleton configuration loader
- **config.json**: Provider selection, agent behavior, and model settings
- **.env**: API keys, database connection strings (not in version control)

## Key Design Patterns

### Provider Factory Pattern

The LLM factory uses a configuration-driven provider selection. Set the `provider` field in config.json to switch between OpenAI, Azure, LM Studio, or Ollama. Each provider is implemented as a separate class (llm_provider_*.py).

### Agent Safeguards

The AgentRunner enforces multiple safety limits:
- `timeout_seconds`: Maximum execution time
- `max_tool_calls`: Maximum number of tool invocations  
- `recursion_limit`: Maximum LangGraph steps
- `max_iterations`: Additional iteration limit

### Streaming Output

The application uses LangGraph's native streaming capabilities. The StreamingOutputFormatter formats execution events for console output.

### Request Flow

1. FastAPI receives POST to `/query` with question and optional history
2. History converted to LangChain messages via utils/agent_utils.py
3. AgentRunner executes with configured safeguards
4. Agent may call SQL toolkit and custom tools during reasoning
5. Final response extracted and returned as JSON

## Configuration Approach

### Provider Selection

Edit config.json and set the `provider` field to one of: `openai`, `azure`, `lmstudio`, `ollama`

### Database Tools

Database tools are enabled by setting `agent.enable_db: true` in config.json and providing DB_* environment variables. The agent will automatically load available tables and expose SQL querying capabilities.

### Environment Variables

The application requires a `.env` file with provider-specific credentials:

**OpenAI**: `OPENAI_API_KEY`
**Azure**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`
**LM Studio**: `LMSTUDIO_BASE_URL` (defaults to http://localhost:1234/v1)
**Database**: `DB_HOST`, `DB_NAME`, optional `DB_USERNAME`/`DB_PASSWORD` or `DB_USE_WINDOWS_AUTH=true`

See ENV_VARIABLES.md for complete reference.

## Examples Directory

The examples/ directory contains executable demos for various patterns:
- main_example.py: Basic agent execution
- streaming_example.py: Streaming responses
- custom_tool_example.py: Adding custom tools (relevant for stock-specific tools)
- multi_agent_example.py: Multiple agent patterns
- self_correction_example.py: Agent self-correction patterns
- human_in_the_loop_example.py: Human-in-the-loop workflows
- few_shot_prompting_example.py: Few-shot prompting patterns
- structured_output_example.py: Structured output generation (useful for financial data)

**For Stock Analysis Development**: Prioritize working with custom_tool_example.py to create stock-specific tools, and structured_output_example.py for properly formatted financial data responses.

## Stock-Focused Development Considerations

When implementing features or making architectural decisions, prioritize:

**Stock Data Access**: The agent should be able to query stock prices, historical data, and financial information through database tools or external APIs.

**Financial Analysis**: Tools should support calculations like portfolio returns, risk metrics, and comparative analysis between stocks or indices.

**Conversation Context**: Stock queries often involve follow-up questions - maintain conversation history to allow multi-step analysis (e.g., "Compare those returns to the S&P 500").

**Risk & Safety**: Even as a demo, implement safeguards around financial advice - the agent should provide analysis, not personalized investment recommendations.

**Real-time vs Historical**: Consider both current market data and historical performance analysis capabilities.

**Portfolio Focus**: Support multi-stock queries and portfolio-level analysis, not just individual stock lookups.

## Development Notes

- The project uses Python 3.13 (specified in .python-version)
- uv is the recommended dependency manager (uv.lock provides pinned dependencies)
- LangServe endpoints are available at `/agent` if langserve is installed
- Prometheus metrics exposed at `http://localhost:9090/metrics` when telemetry is enabled
- CORS is currently permissive for development
- All development should align with the stock chatbot backend purpose

## Testing Strategy

- tests/test_tools.py: Unit tests for individual tools
- tests/test_api.py: API endpoint integration tests  
- Test files use pytest with async support (pytest-asyncio)

## Important Files

**Stock Analysis Development**:
- **messages/system_prompt.txt**: Edit this to align agent behavior with financial analysis focus and risk disclaimers
- **tools.py**: Add stock-specific tools here (price queries, financial metrics, portfolio analysis)
- **table_info.json**: Define database schemas for stock data, company information, and historical prices

**General Configuration**:
- **config.json**: Edit this to change providers, agent behavior, or model settings
- **.env**: Create this with your API keys and database credentials