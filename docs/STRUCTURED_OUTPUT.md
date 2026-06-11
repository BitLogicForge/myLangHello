# Structured Output Guide

## Goal

Describe how structured output can improve this project and document the best practical way to introduce it without making the agent overly rigid.

## What Structured Output Means

Structured output means asking the model to return data in a predefined shape instead of only free-form text.

Example of unstructured output:

```json
{
  "output": "I found 5 countries starting with B."
}
```

Example of structured output:

```json
{
  "status": "success",
  "answer": "I found 5 countries starting with B.",
  "sources": ["sql_db_query"],
  "warnings": [],
  "data": {
    "row_count": 5
  }
}
```

## Why It Matters In This Project

Right now the API mainly returns a single text field from the final model message.

That is simple, but it has limits:

- Harder to render rich UI states
- Harder to test reliably
- Harder to expose citations, warnings, or tool summaries
- Harder to distinguish success, error, and partial-completion cases

Structured output would make this project easier to grow into:

- a stronger demo API
- a frontend-friendly backend
- a more testable agent service
- a safer system for tool-driven behavior

## Important Clarification

Structured output is not something you must build completely by hand.

LangChain supports schema-based structured output, including Pydantic models. However, this project still needs to define:

- what schema to use
- when structured output is required
- how to validate and return it from FastAPI
- what fallback behavior to use if validation fails

So the model-side support exists, but the application contract still needs to be designed.

## Best Use Cases For Structured Output Here

### 1. Final API responses

This is the best first use case.

Why:

- lowest implementation risk
- immediate frontend/API value
- avoids forcing every intermediate step into a schema

### 2. SQL-oriented answers

When the agent uses the SQL toolkit, structured output can expose:

- plain-language answer
- whether the query succeeded
- warnings about assumptions
- row count
- query metadata

### 3. Tool execution summaries

For debugging or observability, the final response can include:

- tools used
- whether each succeeded
- high-level action summary

### 4. Research mode with citations

If fresh web or retrieval-based answers are added later, structured output is a strong fit for:

- answer
- sources
- citations
- confidence notes

## Recommended Design Principles

### Keep the first schema small

Do not try to model every possible agent state in version 1.

Start with a schema that supports:

- answer text
- status
- warnings
- sources
- optional payload data

### Use one schema for the final response first

Do not immediately force structured output across:

- every tool result
- every intermediate reasoning step
- every internal agent event

The first win is at the API boundary.

### Support all states explicitly

A good schema should cover:

- success
- partial success
- error
- empty result

### Keep text plus machine-readable fields

The response should still include a human-readable `answer`, even if it also contains structured metadata.

That gives you:

- readable logs
- better demos
- easier manual testing

## Best First Schema For This Repo

This is the most balanced starting point:

```json
{
  "status": "success",
  "answer": "Found 5 countries starting with B.",
  "sources": ["sql_db_query"],
  "warnings": [],
  "data": {
    "row_count": 5
  },
  "suggested_follow_up": []
}
```

Recommended fields:

- `status`: `success`, `partial`, `error`
- `answer`: plain-language final response
- `sources`: tool names, data sources, or retrieval sources
- `warnings`: assumptions, limitations, or safety notes
- `data`: optional structured payload
- `suggested_follow_up`: optional next questions

## Suggested Pydantic Model

```python
from typing import Any, Literal
from pydantic import BaseModel, Field


class AgentStructuredResponse(BaseModel):
    status: Literal["success", "partial", "error"] = "success"
    answer: str = Field(..., description="Final user-facing response")
    sources: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)
    suggested_follow_up: list[str] = Field(default_factory=list)
```

## Where It Fits In This Codebase

The cleanest integration points are:

- `models/api_models.py`
  - add the structured response model
- `routes/agent_routes.py`
  - validate and return structured output
- `services/agent_runner.py`
  - optionally capture extra metadata for `data` or `sources`

## Best Rollout Plan

### Phase 1: Add a structured final response model

Keep the current agent behavior mostly unchanged and introduce a new final response contract.

Good target:

- preserve `session_id`
- replace plain `output` with structured fields
- add fallback handling

### Phase 2: Add structured SQL result metadata

Expose useful metadata when SQL is involved, such as:

- row count
- whether results were empty
- whether the model made assumptions

### Phase 3: Add citations and source-level details

Especially valuable once retrieval or web-grounded answers exist.

### Phase 4: Optional mode-based schemas

Only after the base shape works well, consider specialized schemas for:

- SQL answers
- research answers
- file actions

## Best Ideas For This Project

### Best idea 1: Keep a stable outer schema

Even if the internal `data` changes by feature, the outer response shape should stay stable.

Example:

- `status`
- `answer`
- `sources`
- `warnings`
- `data`

This reduces frontend and API churn.

### Best idea 2: Use `data` for feature-specific payloads

Do not create many totally different top-level response contracts too early.

Put specialized values under `data`, for example:

```json
{
  "status": "success",
  "answer": "Found 5 rows.",
  "sources": ["sql_db_query"],
  "warnings": [],
  "data": {
    "row_count": 5,
    "table_names": ["Countries"]
  }
}
```

### Best idea 3: Keep warnings first-class

Warnings are often as important as the answer in agent systems.

Examples:

- query required assumptions
- no exact result found
- external source may be stale
- tool failed and answer used fallback reasoning

### Best idea 4: Preserve a readable answer

Do not optimize only for machines. This is still an interactive agent.

Every structured response should keep:

- a concise answer
- a clear status
- optional details

### Best idea 5: Validate, then fallback gracefully

If structured parsing fails:

- do not crash the whole request
- return a safe fallback response
- record the failure in logs or telemetry

Example fallback:

```json
{
  "status": "partial",
  "answer": "The agent produced a response, but structured parsing failed.",
  "sources": [],
  "warnings": ["Structured output validation failed"],
  "data": {},
  "suggested_follow_up": []
}
```

## Things To Avoid

### Avoid one giant schema for everything

This usually becomes hard to maintain and difficult for the model to satisfy reliably.

### Avoid removing plain-language answers

Machine-readable data is not a substitute for a clear user-facing answer.

### Avoid strictness too early

If the schema is too detailed at the start, the model may become less natural or fail validation more often.

### Avoid modeling internal chain-of-thought

Structured output should describe useful results and metadata, not hidden reasoning traces.

## Example Response Shapes

### Success

```json
{
  "status": "success",
  "answer": "Found 5 matching countries.",
  "sources": ["sql_db_query"],
  "warnings": [],
  "data": {
    "row_count": 5
  },
  "suggested_follow_up": ["Do you want the country codes too?"]
}
```

### Partial result

```json
{
  "status": "partial",
  "answer": "I found some data, but one external tool failed.",
  "sources": ["sql_db_query", "http_get"],
  "warnings": ["Weather lookup failed for one result"],
  "data": {
    "row_count": 5
  },
  "suggested_follow_up": []
}
```

### Error

```json
{
  "status": "error",
  "answer": "I could not complete the request.",
  "sources": [],
  "warnings": ["Database connection failed"],
  "data": {},
  "suggested_follow_up": ["Try again later or verify database settings."]
}
```

## Testing Recommendations

When this feature is implemented, test at least:

- valid structured success response
- empty result response
- partial result response
- error response
- invalid model output fallback

## Recommended Next Step

The best next implementation step is:

1. Add a new Pydantic response model in `models/api_models.py`.
2. Update `/query` to return the structured final response.
3. Keep `answer` human-readable.
4. Log and handle validation failures safely.

## Summary

Structured output is one of the highest-value upgrades for this project because it improves reliability without requiring a full architectural rewrite.

The best approach is:

- start small
- keep one stable outer schema
- preserve readable answers
- include warnings and sources
- validate and fallback safely
