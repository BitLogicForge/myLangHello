# MyLangHello

A small FastAPI + LangGraph demo project for running a tool-using chat agent with SQL access, configurable LLM providers, and basic telemetry.

## What This Project Does

This project exposes an agent that can:

- Answer natural language questions
- Use LangChain/LangGraph tools during reasoning
- Query a SQL Server database through the LangChain SQL toolkit
- Accept optional conversation history
- Run through a FastAPI API
- Stream agent execution through LangServe when available
- Publish Prometheus metrics for local monitoring

## Main Capabilities

- Provider selection through `config.json`
- OpenAI, Azure OpenAI, and LM Studio support
- Config-driven runtime guardrails
- SQL database toolkit integration
- Custom utility tools such as calculator, file read/write, HTTP GET, date, and demo helpers
- Health and config endpoints
- Optional LangServe playground and streaming endpoints

## Architecture Overview

The project is organized around a few core layers:

### API layer

- `api.py`: FastAPI app, middleware, route registration, startup wiring
- `routes/agent_routes.py`: main `/query` endpoint
- `routes/health_routes.py`: health endpoint
- `routes/config_routes.py`: basic config endpoint

### Agent orchestration

- `main.py`: defines the core `AgentApp` class (orchestrating configuration, runner, and formatter)
- `services/agent_configurator.py`: builds the agent and dependencies
- `services/agent_factory.py`: creates the LangGraph/LangChain agent
- `services/agent_runner.py`: enforces runtime guardrails such as timeout and tool-call limits

### Model and prompt setup

- `services/llm_factory.py`: selects provider from config
- `services/llm_provider_openai.py`
- `services/llm_provider_azure.py`
- `services/llm_provider_lmstudio.py`
- `services/prompt_builder.py`: loads the system prompt
- `messages/system_prompt.txt`: system prompt text

### Tools and utilities

- `services/tools_manager.py`: registers custom tools
- `tools.py`: custom tool implementations
- `utils/agent_utils.py`: converts request history to LangChain messages
- `services/output_formatter.py`: console-friendly streaming trace formatting
- `services/telemetry.py`: Prometheus metrics support

### Models and config

- `models/api_models.py`: request and response schemas
- `config.py`: singleton config loader
- `config.json`: provider and runtime settings
- `ENV_VARIABLES.md`: expected environment variables

## Request Flow

1. FastAPI receives a request on `/query`.
2. Request history is converted into LangChain messages.
3. `AgentRunner` executes the agent with configured safeguards.
4. The LangGraph agent may call SQL toolkit tools and custom tools.
5. The final message is extracted and returned as the API response.
6. Telemetry records request-level metrics when enabled.

## Requirements

- Python 3.13 (configured in `.python-version` and `pyproject.toml`)
- Access to a configured LLM provider:
  - OpenAI
  - Azure OpenAI
  - LM Studio
- SQL Server access if you want database tooling enabled
- A `.env` file with the required secrets and connection values

## Installation

### Using `uv` (Recommended)
This project is configured with `uv` for dependency management:
```bash
# Setup the virtual environment and install all pinned dependencies
uv sync
```

### Using standard pip
```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

## Configuration

### 1. Choose provider

Set the `provider` field in `config.json`:

- `openai`
- `azure`
- `lmstudio`

### 2. Add environment variables

Use `ENV_VARIABLES.md` as the reference for required variables.

Typical `.env` values include:

```env
OPENAI_API_KEY=your-key
DB_HOST=localhost
DB_NAME=YourDatabase
DB_USERNAME=your-user
DB_PASSWORD=your-password
DB_DRIVER=ODBC Driver 17 for SQL Server
```

For LM Studio, a typical local setup is:

```env
LMSTUDIO_BASE_URL=http://localhost:1234/v1
LMSTUDIO_API_KEY=lm-studio
```

### 3. Configure agent behavior

Important settings live in `config.json`:

- `agent.recursion_limit`
- `agent.timeout_seconds`
- `agent.max_tool_calls`
- `agent.sql.include_tables`
- provider-specific model settings

## Running The Project

Run the API server:

```bash
uv run python api.py
# Or with standard python: python api.py
```

Run the local console test app:

```bash
uv run python examples/main_example.py
# Or with standard python: python examples/main_example.py
```

## API Endpoints

### `GET /`

Returns service metadata and useful links.

### `GET /health`

Returns basic service and agent status.

### `GET /config`

Returns basic model and tool configuration details.

### `POST /query`

Primary request endpoint when using the manual API path.

Example request:

```json
{
  "question": "List the first 5 countries starting with B from the database",
  "session_id": "demo-1",
  "history": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi, how can I help?" }
  ]
}
```

Example response:

```json
{
  "output": "Agent response text",
  "session_id": "demo-1"
}
```

## LangServe Support

If `langserve` is installed and the agent loads correctly, the app also exposes LangServe routes under `/agent`.

Useful routes include:

- `/agent/playground`
- `/agent/stream`

## Telemetry

The project includes local Prometheus metrics support through `services/telemetry.py`.

When telemetry starts successfully, metrics are exposed at:

```text
http://localhost:9090/metrics
```

Tracked categories include:

- request counts and durations
- LLM call counts and token totals
- agent iteration counts
- tool call counts and durations
- database query metrics

## Included Tools

The current custom tool set includes:

- calculator
- weather
- read file
- write file
- HTTP GET
- random joke
- current date
- loan calculator
- currency converter
- joke formatting

Note: some tools are intentionally demo-oriented rather than production-grade.

## Project Structure

```text
.
|-- api.py                    # FastAPI application entry point
|-- main.py                   # Library entry point defining AgentApp orchestrator
|-- config.py                 # Singleton config loader
|-- config.json               # Config provider and runtime settings
|-- tools.py                  # Custom tool definitions
|-- pyproject.toml            # Project dependencies and configuration
|-- uv.lock                   # Pinned dependency lockfile
|-- .python-version           # Pinned python version (3.13)
|-- routes/                   # API routes (agent, health, config)
|-- services/                 # Agent orchestration, LLM factories, output, telemetry
|-- models/                   # Request/Response schemas
|-- utils/                    # Utility scripts (logging, history messages)
|-- messages/                 # Prompt definitions
|-- examples/                 # Executable examples (main_example.py, streaming_example.py)
|-- ENV_VARIABLES.md          # Environment variables reference
|-- DOCUMENTATION_IDEAS.md    # Ideas for future documentation
```

## Known Limitations

- Some tools are simulated demo tools rather than grounded external integrations
- Session IDs are accepted, but durable memory is not yet fully implemented
- The project is suitable for local/demo use and still needs hardening for production
- CORS is currently permissive

## Next Documentation To Add

Good follow-up docs for this repo would be:

- deployment guide
- testing guide
- architecture decision notes
- provider-specific setup examples
- database safety and tool usage policy

## Related Docs

- [ENV_VARIABLES.md](./ENV_VARIABLES.md)
- [DOCUMENTATION_IDEAS.md](./DOCUMENTATION_IDEAS.md)
