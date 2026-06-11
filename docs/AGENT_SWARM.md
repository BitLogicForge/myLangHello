# Agent Swarm Notes

## Goal

Describe how an agent swarm differs from the current multi-agent discussion mode in this project, and document whether it is a good future direction.

## Short Answer

The current implementation is **not a swarm**.

The current implementation is a **bounded discussion council**:

- fixed set of personas
- fixed speaking order
- fixed number of rounds
- one final moderator synthesis
- no autonomous delegation
- no dynamic routing

An **agent swarm** is more open-ended and autonomous:

- agents can decide which other agents should act next
- agents may spawn or route work dynamically
- work can branch, merge, retry, and reassign
- there is usually shared state or a richer coordination mechanism
- the system behaves more like a distributed team than a staged panel discussion

## What We Have Today

Current discussion mode in `services/discussion_orchestrator.py` works like this:

1. Build a shared context from the user question and optional history.
2. Ask each persona to respond in sequence.
3. Repeat for a small fixed number of rounds.
4. Ask the moderator to summarize the discussion.
5. Return the transcript and final answer.

This is intentionally simple and predictable.

## Why The Current Approach Is Good

For this repo, the current council approach is strong because it is:

- easy to explain in a demo
- deterministic enough to debug
- low-risk compared to autonomous systems
- easy to log and visualize
- compatible with the current single-request API model

It gives the “multiple agents talking” experience without introducing large orchestration complexity.

## What An Agent Swarm Usually Means

In practice, a swarm usually has some combination of these capabilities:

- dynamic task routing
- specialist agents with different tools
- shared memory or task state
- agents deciding whether to continue, escalate, or stop
- optional parallel execution
- retry and recovery behavior
- structured intermediate outputs

Instead of:

- `Agent A -> Agent B -> Agent C -> Moderator`

you get something more like:

- planner receives task
- planner delegates to researcher
- researcher asks database agent
- researcher escalates to web agent
- reviewer checks result
- planner revises plan
- final synthesizer responds

The key difference is that the path is not predetermined.

## Current Council vs Future Swarm

### Current council

- fixed personas
- same LLM instance and same general prompt base
- ordered loop
- no branching
- no real task ownership
- transcript is the main artifact

### Future swarm

- different agent roles with real responsibilities
- dynamic handoff between agents
- richer state object than a plain transcript
- tool access can vary by agent
- can support planner / worker / reviewer patterns
- needs stronger stopping and safety controls

## The Biggest Differences

### 1. Coordination model

Current approach:

- the orchestrator controls everything

Swarm approach:

- agents influence the workflow itself

### 2. State management

Current approach:

- state is mostly just transcript text

Swarm approach:

- state is usually structured
- tasks, subtasks, status, artifacts, and handoff decisions matter

### 3. Tool specialization

Current approach:

- personas are mostly style/personality differences

Swarm approach:

- each agent often owns a real function
- example: researcher, SQL agent, critic, formatter, planner

### 4. Reliability

Current approach:

- easier to debug and cap with fixed rounds

Swarm approach:

- more capable, but easier to make unstable
- can loop, duplicate work, or drift without clear rules

## Why A Swarm Is Not The Best Next Step Here

This project is not yet ready for a true swarm as the next upgrade.

Main reasons:

- no durable shared state yet
- limited structured output
- current request/response model is still simple
- tools are not yet partitioned by agent role
- current logging is transcript-oriented, not task-graph-oriented

If a swarm is added too early, the likely result is:

- more latency
- more chaos
- harder debugging
- less trustworthy outputs

## When A Swarm Would Make Sense

A swarm becomes more realistic after these foundations exist:

1. Structured outputs for intermediate steps
2. Persistent state or checkpointing
3. Better tool boundaries
4. Clear stop conditions
5. Evaluation coverage for orchestration behavior

Once those exist, this project could support more serious multi-agent workflows.

## Effort Assessment

Short version:

- **Basic swarm prototype**: medium effort
- **Reliable swarm for demos**: medium-high effort
- **Production-style swarm**: high effort

Rough estimate for this repo:

- **1 to 2 days**
  - very basic planner + specialist handoff
  - minimal state
  - low reliability
  - good for experimentation only

- **3 to 5 days**
  - clearer agent roles
  - bounded handoffs
  - improved logging
  - structured intermediate state
  - realistic target for a stronger internal demo

- **1 to 2+ weeks**
  - durable state
  - robust stopping logic
  - better validation and evals
  - safer tool boundaries
  - much closer to something you can trust repeatedly

Main cost drivers:

- state management
- orchestration rules
- debugging handoffs
- preventing loops and duplicated work
- testing non-deterministic behavior

For this codebase specifically, switching from the current council to a true swarm is **not a tiny refactor**. It is feasible, but it becomes much easier after structured output and persistence are in place.

## Recommended Swarm Shape For This Repo

If swarm behavior is added later, the best pattern is not “random agents arguing.”

The best pattern is a small role-based workflow such as:

- `Planner`
- `Researcher`
- `SQL Analyst`
- `Reviewer`
- `Formatter`

Example:

1. Planner interprets the user task.
2. Planner decides whether database access, web lookup, or file analysis is needed.
3. Specialist agent performs the work.
4. Reviewer checks quality and risk.
5. Formatter produces the final answer or structured payload.

This would feel much more useful than a pure personality-based swarm.

## Example Progression Path

### Phase 1

Keep the current council mode for fun demos.

### Phase 2

Add structured outputs and stronger state handling.

### Phase 3

Introduce role-based specialist agents with limited tool access.

### Phase 4

Allow dynamic handoff between planner and specialists.

### Phase 5

Optionally add parallel work for independent subtasks.

## Risks In A Swarm Design

The main risks are:

- infinite or wasteful loops
- duplicated work
- conflicting agent conclusions
- poor stopping logic
- hard-to-read logs
- rising token and latency costs

To control those risks, a future swarm should include:

- maximum steps
- maximum handoffs
- role-specific tool access
- structured state
- logging per task transition

## Recommendation

For now:

- keep the current council mode as the “fun multi-agent demo”

For the future:

- treat swarm architecture as a second-stage upgrade after structured output and persistence

That order gives a much better chance of ending up with something impressive and maintainable.

## Summary

The current implementation is a **scripted multi-agent council**.

A swarm would be a **dynamic, autonomous, task-routing multi-agent system**.

The council is the right fit for now.
The swarm is a good future direction, but only after the project gains:

- structured state
- durable memory
- clearer agent roles
- stronger orchestration controls
