# Documentation Ideas

## Goal

Capture practical, state-of-the-art capability ideas for this demo project so future work stays focused on the highest-value upgrades.

## Current Foundation

The project already has a solid demo base:

- Configurable LLM providers
- LangGraph/LangChain agent wiring
- SQL toolkit integration
- FastAPI API surface
- Streaming output formatting
- Telemetry hooks

That means the best next step is not adding random tools, but improving trustworthiness, grounding, persistence, and product quality.

## Highest-Value SOTA Ideas

### 1. Persistent memory and resumable threads

Use LangGraph persistence/checkpointing so conversations can continue across requests and recover after failures.

Why it matters:

- Real session memory instead of request-only history passing
- Better demo story for long-running workflows
- Foundation for human-in-the-loop approvals
- More reliable stateful agent behavior

Best fit in this codebase:

- `services/agent_factory.py`
- `routes/agent_routes.py`
- `main.py`

### 2. Grounded retrieval over local docs and schema

Add retrieval over project documents, table descriptions, and business notes so answers are based on known sources, not only model reasoning.

Why it matters:

- Better answers for internal knowledge questions
- Stronger SQL generation context
- Easier to explain and trust agent responses
- More realistic than purely prompt-based behavior

Good candidate sources:

- Database/table metadata
- Internal documentation
- Examples and usage notes
- Curated schema descriptions

### 3. Web-grounded research mode with citations

Add a research mode that can retrieve fresh information and return citations, instead of relying on a raw generic HTTP tool.

Why it matters:

- Enables current-event and live-data use cases
- Makes outputs more transparent and demo-friendly
- Stronger trust model than unstructured browsing

### 4. MCP-based integrations

Adopt Model Context Protocol support so the demo can connect cleanly to external tools and data sources without hardcoding every integration.

Why it matters:

- Modern integration story
- Easier extensibility
- Cleaner boundary between agent logic and external capabilities
- Better long-term architecture than ad hoc custom tools

### 5. Structured outputs for reliable downstream use

Return validated JSON shapes for plans, citations, SQL intents, and action results instead of only plain-text output.

Why it matters:

- Easier frontend rendering
- Easier testing
- Easier automation
- Less brittle parsing

Examples:

- Final answer + citations
- SQL query plan + safety notes
- Tool execution summary
- Error state payloads

### 6. Human approval for risky actions

Require approval before file writes, sensitive reads, or other higher-risk actions.

Why it matters:

- Safer demo behavior
- Better enterprise story
- Stronger operational trust
- Good fit for stateful agent workflows

### 7. Evaluation and regression testing

Add an eval harness for prompt behavior, tool selection, SQL correctness, and failure handling.

Why it matters:

- Prevents regressions as prompts and tools evolve
- Makes demos repeatable
- Gives confidence before changing agent behavior

Recommended coverage:

- Happy path
- Empty results
- Tool failure
- Timeout behavior
- Bad SQL or invalid assumptions

## Important Gaps To Fix First

These are worth addressing before chasing more advanced features:

### Simulated tools reduce credibility

Some current tools are intentionally fake or demo-only, especially weather and currency conversion. Replacing those with one or two real, grounded capabilities will improve the project more than adding many new tools.

### Session support is partial

`session_id` exists in the API contract, but agent state is not truly persisted across runs yet.

### Config exposure is incomplete

The `/config` route expects fields that are not clearly exposed from the main app object, so config introspection may not reflect reality.

### Security posture is demo-only

Open CORS and unrestricted file tools are acceptable for local development, but they weaken the story for a secure production-style demo.

## Recommended Implementation Order

1. Add persistent conversation threads with checkpointing.
2. Replace fake tools with one real retrieval or web-grounded capability.
3. Add approval gates for sensitive actions.
4. Add structured JSON outputs.
5. Add evaluation coverage and regression tests.

## Best Next Feature

If only one feature should be built next, the highest-value option is:

**Persistent SQL copilot with retrieval and citations**

Why this is the best fit:

- It builds directly on the current SQL agent design
- It feels modern and useful immediately
- It improves trust, not just complexity
- It is more compelling than adding multi-agent behavior too early

## Out of Scope For Now

These ideas may be interesting later, but should not be the first investment:

- Multi-agent orchestration before grounding and memory exist
- Large numbers of novelty tools without real data backing
- UI polish before response quality and trust improve

## Suggestion Log

💡 Suggestion: Focus on making one agent grounded, resumable, and trustworthy before exploring multi-agent workflows. This will produce a much stronger demo with less complexity and lower regression risk.
