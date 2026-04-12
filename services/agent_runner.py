"""Runtime helpers for executing the agent with config-driven safeguards."""

import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

from config import Config


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

        timeout_seconds = float(timeout_value) if timeout_value not in (None, 0) else None
        max_tool_calls = (
            int(max_tool_calls_value) if max_tool_calls_value not in (None, 0) else None
        )

        return cls(
            recursion_limit=int(config.get("agent.recursion_limit", 15)),
            timeout_seconds=timeout_seconds,
            max_tool_calls=max_tool_calls,
        )


class AgentRunner:
    """Execute the agent while enforcing runtime guardrails."""

    def __init__(self, agent_executor: Any, settings: AgentExecutionSettings):
        self.agent_executor = agent_executor
        self.settings = settings

    def run(
        self,
        agent_input: dict[str, Any],
        on_event: Optional[Callable[[dict[str, Any], int], None]] = None,
    ) -> Optional[dict[str, Any]]:
        """Run the agent through streaming and return an invoke-shaped response."""
        run_config = {"recursion_limit": self.settings.recursion_limit}
        start_time = time.monotonic()
        tool_call_count = 0
        step_count = 0
        final_messages: list[Any] = []

        for event in self.agent_executor.stream(agent_input, config=run_config):
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
    def _count_tool_calls(event: dict[str, Any]) -> int:
        """Count tool calls in a streamed event."""
        tool_calls = 0
        for node_data in event.values():
            messages = node_data.get("messages", [])
            for msg in messages:
                calls = getattr(msg, "tool_calls", None)
                if calls:
                    tool_calls += len(calls)
        return tool_calls

    @staticmethod
    def _extract_latest_messages(event: dict[str, Any]) -> list[Any]:
        """Extract the latest messages payload from a streamed event."""
        latest_messages: list[Any] = []
        for node_data in event.values():
            messages = node_data.get("messages", [])
            if messages:
                latest_messages = messages
        return latest_messages
