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
- GitHub MCP integration available for repository and PR workflows
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

- `main.py`: top-level app orchestrator
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

- Python 3.11+ recommended
- Access to a configured LLM provider:
  - OpenAI
  - Azure OpenAI
  - LM Studio
- SQL Server access if you want database tooling enabled
- A `.env` file with the required secrets and connection values

## Installation

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
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

When running inside the devcontainer, `localhost` means the container itself. The devcontainer sets:

```env
LMSTUDIO_BASE_URL=http://host.docker.internal:1234/v1
```

Keep LM Studio's local server running on the host and make sure the port is allowed through your firewall.

### 3. Configure agent behavior

Important settings live in `config.json`:

- `agent.recursion_limit`
- `agent.timeout_seconds`
- `agent.max_tool_calls`
- `agent.sql.include_tables`
- provider-specific model settings

## Running The Project

Run the API server:

```powershell
python api.py
```

Run the local console app:

```powershell
python main.py
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
  "session_id": "demo-1",
  "mode": "single"
}
```

### Discussion mode

For a fun multi-agent demo, `/query` also supports a bounded discussion mode with three personas plus a moderator summary.

Example request:

```json
{
  "question": "Should we add a dashboard to this project?",
  "mode": "discussion",
  "discussion_rounds": 2,
  "include_discussion_transcript": true
}
```

Example response shape:

```json
{
  "output": "Final moderator summary...",
  "session_id": null,
  "mode": "discussion",
  "transcript": [
    {
      "round_number": 1,
      "speaker": "Spark",
      "role": "idea generator",
      "content": "..."
    }
  ],
  "participants": ["Spark", "Shield", "Forge", "Moderator"]
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
|-- api.py
|-- main.py
|-- config.py
|-- config.json
|-- tools.py
|-- routes/
|-- services/
|-- models/
|-- utils/
|-- messages/
|-- examples/
|-- ENV_VARIABLES.md
|-- docs/
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
- [Documentation Ideas](./docs/DOCUMENTATION_IDEAS.md)
- [Agent Swarm Notes](./docs/AGENT_SWARM.md)
- [Structured Output Guide](./docs/STRUCTURED_OUTPUT.md)
