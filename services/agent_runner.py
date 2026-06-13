"""Runtime helpers for executing the agent with config-driven safeguards."""

import time
from dataclasses import dataclass
from typing import Callable, cast

from config import Config
from .agent_configurator import SupportsAStream


# MARK: Settings Schema
@dataclass(frozen=True)
class AgentExecutionSettings:
    """Runtime settings enforced for each agent request."""

    recursion_limit: int
    timeout_seconds: float | None
    max_tool_calls: int | None

    @classmethod
    def from_config(cls, config: Config) -> "AgentExecutionSettings":
        """Build execution settings from config."""
        timeout_value = config.get("agent.timeout_seconds", 60)
        max_tool_calls_value = config.get("agent.max_tool_calls", 10)
        recursion_limit_value = config.get("agent.recursion_limit", 15)

        timeout_seconds = (
            float(timeout_value) if isinstance(timeout_value, (int, float, str)) else None
        )
        max_tool_calls = (
            int(max_tool_calls_value) if isinstance(max_tool_calls_value, (int, str)) else None
        )
        recursion_limit = (
            int(recursion_limit_value) if isinstance(recursion_limit_value, (int, str)) else 15
        )

        return cls(
            recursion_limit=recursion_limit,
            timeout_seconds=timeout_seconds,
            max_tool_calls=max_tool_calls,
        )


# MARK: Agent Runner
class AgentRunner:
    """Execute the agent while enforcing runtime guardrails."""

    agent_executor: SupportsAStream
    settings: AgentExecutionSettings

    def __init__(self, agent_executor: SupportsAStream, settings: AgentExecutionSettings):
        self.agent_executor = agent_executor
        self.settings = settings

    async def run(
        self,
        agent_input: dict[str, object],
        on_event: Callable[[dict[str, object], int], None] | None = None,
    ) -> dict[str, object] | None:
        """Run the agent through streaming and return an invoke-shaped response."""
        run_config: dict[str, object] = {"recursion_limit": self.settings.recursion_limit}
        start_time = time.monotonic()
        tool_call_count = 0
        step_count = 0
        final_messages: list[object] = []

        # Use async streaming (astream) to support async tool invocation
        async for event in self.agent_executor.astream(agent_input, config=run_config):
            step_count += 1
            self._enforce_timeout(start_time)

            tool_call_count += self._count_tool_calls(event)
            self._enforce_tool_call_limit(tool_call_count)

            latest_messages = self._extract_latest_messages(event)
            if latest_messages:
                final_messages = latest_messages

            if on_event is not None:
                on_event(event, step_count)

        self._enforce_timeout(start_time)

        if not final_messages:
            return None

        return {"messages": final_messages}

# MARK: Safeguards
    def _enforce_timeout(self, start_time: float) -> None:
        """Stop execution once the configured timeout is exceeded."""
        timeout_seconds = self.settings.timeout_seconds
        if timeout_seconds is None:
            return

        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(f"Agent execution exceeded timeout of {timeout_seconds:.1f} seconds")

    def _enforce_tool_call_limit(self, tool_call_count: int) -> None:
        """Stop execution once the configured tool-call limit is exceeded."""
        max_tool_calls = self.settings.max_tool_calls
        if max_tool_calls is None:
            return

        if tool_call_count > max_tool_calls:
            raise RuntimeError(f"Agent exceeded max_tool_calls limit of {max_tool_calls}")

    @staticmethod
    def _count_tool_calls(event: dict[str, object]) -> int:
        """Count tool calls in a streamed event."""
        tool_calls = 0
        for node_data in event.values():
            if isinstance(node_data, dict):
                node_dict: dict[str, object] = cast(dict[str, object], node_data)
                messages = cast(list[object], node_dict.get("messages", []))
                for msg in messages:
                    calls = cast(list[object] | None, getattr(msg, "tool_calls", None))
                    if isinstance(calls, list):
                        tool_calls += len(calls)
        return tool_calls

    @staticmethod
    def _extract_latest_messages(event: dict[str, object]) -> list[object]:
        """Extract the latest messages payload from a streamed event."""
        latest_messages: list[object] = []
        for node_data in event.values():
            if isinstance(node_data, dict):
                node_dict: dict[str, object] = cast(dict[str, object], node_data)
                messages = cast(list[object], node_dict.get("messages", []))
                if messages:
                    latest_messages = messages
        return latest_messages
