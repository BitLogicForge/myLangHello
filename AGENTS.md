# AGENTS.md

## Project Mission

Build and maintain a fast, secure, and clean FastAPI + LangGraph agent service that is reliable for demos and easy to evolve.

## Stack And Scope

- Backend: FastAPI, Pydantic, Uvicorn
- Agent: LangChain/LangGraph with provider abstraction (OpenAI, Azure, LM Studio)
- Tools: SQL toolkit + custom tools in `tools.py`
- Config: `config.json` + environment variables from `ENV_VARIABLES.md`
- Observability: logging + telemetry hooks

Do not introduce unrelated architecture changes unless explicitly requested.

## Mandatory Workflow For Every Task

Before doing work, always provide:

1. Goal restatement
2. Assumptions
3. Plan
4. Risks

If requirements are truly unclear, ask one focused question. Do not guess.

## Implementation Rules

- Prefer explicit, readable code over clever shortcuts.
- Keep changes small and scoped to the requested outcome.
- Preserve existing behavior unless change is requested.
- Handle all states for user-facing/API behavior:
  - success
  - loading/in-progress when relevant
  - empty result
  - error and fallback
- Keep public contracts stable (request/response models and route behavior).

## API And Backend Standards

- Validate all external input server-side with Pydantic models.
- Return clear HTTP status codes and actionable error messages.
- Keep route handlers thin; move business logic to `services/`.
- Maintain separation of concerns:
  - `routes/` for transport
  - `services/` for orchestration/business logic
  - `models/` for schemas
  - `utils/` for shared helpers

## Security Requirements

- Never hardcode secrets; use env vars only.
- Never log API keys, DB passwords, or sensitive tokens.
- Use parameterized queries and safe database access patterns only.
- Validate file and tool inputs before execution.
- Keep CORS and auth settings environment-appropriate.
- Prefer secure defaults for cookies/headers when web auth is introduced.

## LLM And Tooling Safety

- Enforce and respect runtime guards (timeouts, recursion/tool-call limits).
- Track and surface tool failures clearly; do not silently ignore failures.
- For fallback behavior, fail safely and return useful user-facing messages.
- Keep prompts and provider wiring configurable, not hardcoded per environment.

## Testing And Verification

For logic changes, add or update tests when needed:

- Cover happy path
- Cover edge cases
- Cover failure/error behavior

At minimum before handoff:

1. Run relevant tests for touched areas.
2. Run a quick manual API sanity check (`/health`, changed endpoints).
3. Verify no obvious regressions in existing routes.

If test tooling is missing, note what was validated manually and what remains unverified.

## Documentation Discipline

- Update docs in the same change when behavior/contracts/config change.
- Keep README and docs paths accurate.
- Keep examples aligned with current request/response shapes.
- GitHub MCP is available in this project context and should be used for GitHub operations (PRs, issues, comments, metadata, and commits) when possible.
- Record ideas outside scope as:
  - `Suggestion: <what + why>`

Do not implement out-of-scope ideas without explicit approval.

## Git Hygiene

- Make small, atomic commits.
- One concern per commit.
- Use clear commit messages that explain intent and impact.
- Prefer GitHub MCP tools for remote GitHub workflows; use local git as fallback if MCP auth/capability is blocked.
- Do not rewrite history unless explicitly requested.

## Definition Of Done

A task is done only when it is shippable:

- secure by default for this scope
- error-handled
- no known regression introduced
- docs updated for behavior/config changes
- self-reviewed for clarity and maintainability
