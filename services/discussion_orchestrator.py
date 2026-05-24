"""Lightweight multi-agent discussion orchestrator."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from models.api_models import DiscussionTurn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentPersona:
    """Definition for a single discussion participant."""

    name: str
    role: str
    instruction: str
    allowed_tools: list[str] = None


class DiscussionOrchestrator:
    """Run a bounded discussion between three persona agents and a moderator."""

    DEFAULT_PERSONAS = (
        AgentPersona(
            name="Sterling",
            role="business operator",
            instruction=(
                "You are Sterling, a business-minded operator. Think in terms of leverage, delivery risk, "
                "ROI, stakeholder perception, and what can realistically ship fast. Be crisp, practical, "
                "and commercially minded."
            ),
            allowed_tools=["loan_calculator", "currency_converter"],
        ),
        AgentPersona(
            name="Riot",
            role="chaotic trend chaser",
            instruction=(
                "You are Riot, an impulsive internet-brained trend chaser. You care about virality, vibes, "
                "memorable moments, and what would make people say 'wait, that is actually cool.' You may "
                "suggest risky or slightly absurd ideas, but keep them usable for a product discussion."
            ),
            allowed_tools=["random_joke", "weather"],
        ),
        AgentPersona(
            name="Forge",
            role="database researcher",
            instruction=(
                "You are Forge, a data specialist with direct database access. Query the database to retrieve "
                "actual facts and figures whenever needed. Always base your responses on real schema and table queries."
            ),
            allowed_tools=["sql_db_query", "sql_db_schema"],
        ),
    )

    def __init__(self, llm: BaseChatModel, base_system_prompt: str):
        self.llm = llm
        self.base_system_prompt = base_system_prompt.strip()

    def run(
        self,
        question: str,
        rounds: int = 2,
        history: list[tuple[str, str]] | None = None,
    ) -> dict[str, object]:
        """Run the multi-agent discussion and return transcript plus final summary."""
        bounded_rounds = max(1, min(rounds, 5))
        transcript: list[DiscussionTurn] = []
        conversation_context = self._build_conversation_context(question, history)

        logger.info("Starting discussion mode with %s rounds", bounded_rounds)

        for round_number in range(1, bounded_rounds + 1):
            logger.info("Discussion round %s started", round_number)
            for persona in self.DEFAULT_PERSONAS:
                prompt_messages = self._build_persona_messages(
                    persona=persona,
                    question=question,
                    conversation_context=conversation_context,
                    transcript=transcript,
                    round_number=round_number,
                )
                content = self._invoke_text(prompt_messages, persona=persona)
                self._log_turn(round_number, persona.name, persona.role, content)
                transcript.append(
                    DiscussionTurn(
                        round_number=round_number,
                        speaker=persona.name,
                        role=persona.role,
                        content=content,
                    )
                )

        final_summary = self._build_final_summary(question, conversation_context, transcript)
        self._log_turn(bounded_rounds, "Moderator", "final synthesizer", final_summary)

        return {
            "output": final_summary,
            "transcript": transcript,
            "participants": [persona.name for persona in self.DEFAULT_PERSONAS] + ["Moderator"],
        }

    def _build_conversation_context(
        self,
        question: str,
        history: list[tuple[str, str]] | None,
    ) -> str:
        """Flatten request history into readable context for discussion prompts."""
        lines: list[str] = [f"User request: {question}"]
        if history:
            lines.append("Prior conversation:")
            for role, content in history:
                lines.append(f"- {role}: {content}")
        return "\n".join(lines)

    def _build_persona_messages(
        self,
        persona: AgentPersona,
        question: str,
        conversation_context: str,
        transcript: list[DiscussionTurn],
        round_number: int,
    ) -> list[BaseMessage]:
        """Construct prompt messages for a persona turn."""
        transcript_block = self._format_transcript(transcript)
        user_prompt = (
            f"{conversation_context}\n\n"
            f"Discussion round: {round_number}\n"
            "Current transcript so far:\n"
            f"{transcript_block}\n\n"
            "Respond with a short contribution that helps the group answer the user."
        )

        return [
            SystemMessage(content=self.base_system_prompt),
            SystemMessage(content=persona.instruction),
            HumanMessage(content=user_prompt),
        ]

    def _build_final_summary(
        self,
        question: str,
        conversation_context: str,
        transcript: list[DiscussionTurn],
    ) -> str:
        """Ask a moderator prompt to synthesize the transcript."""
        moderator_prompt = (
            f"{conversation_context}\n\n"
            "You are the moderator. Read the transcript and produce one final answer for the user.\n"
            "Be clear, concise, and reference tradeoffs when useful.\n\n"
            f"Original question: {question}\n\n"
            "Transcript:\n"
            f"{self._format_transcript(transcript)}"
        )

        return self._invoke_text(
            [
                SystemMessage(content=self.base_system_prompt),
                SystemMessage(
                    content=(
                        "You are Moderator, responsible for combining the agents' discussion into one "
                        "helpful final answer."
                    )
                ),
                HumanMessage(content=moderator_prompt),
            ]
        )

    def _invoke_text(self, messages: list[BaseMessage], persona: AgentPersona = None) -> str:
        """Invoke the LLM and normalize the returned text, executing tool calls allowed for the persona if any."""
        # Filter self.tools to only those allowed for this specific persona
        allowed_tools = []
        if self.tools and persona and hasattr(persona, "allowed_tools") and persona.allowed_tools:
            allowed_tools = [t for t in self.tools if t.name in persona.allowed_tools]

        if allowed_tools and hasattr(self.llm, "bind_tools"):
            bound_llm = self.llm.bind_tools(allowed_tools)
            response = bound_llm.invoke(messages)

            # Limit tool execution to a maximum of 3 iterations
            loop_count = 0
            while response.tool_calls and loop_count < 3:
                loop_count += 1
                logger.info(f"Persona {persona.name} triggered {len(response.tool_calls)} tool calls (iteration {loop_count})")

                # Append the assistant's message with tool calls
                messages.append(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    tool_id = tool_call["id"]

                    # Find tool in allowed tools list
                    tool_obj = next((t for t in allowed_tools if t.name == tool_name), None)
                    if tool_obj:
                        try:
                            logger.info(f"Executing tool {tool_name} for persona {persona.name} with args {tool_args}")
                            tool_output = tool_obj.invoke(tool_args)
                        except Exception as e:
                            tool_output = f"Error executing tool {tool_name}: {str(e)}"
                    else:
                        tool_output = f"Error: Tool {tool_name} is not allowed or not found for persona {persona.name}."

                    from langchain_core.messages import ToolMessage
                    messages.append(ToolMessage(content=str(tool_output), tool_call_id=tool_id))

                # Re-invoke LLM with tool outputs in context
                response = bound_llm.invoke(messages)
        else:
            response = self.llm.invoke(messages)

        if isinstance(response, AIMessage):
            content = response.content
        else:
            content = getattr(response, "content", str(response))

        if isinstance(content, list):
            parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            text = "\n".join(part for part in parts if part).strip()
        else:
            text = str(content).strip()

        return text or "No response generated."

    @staticmethod
    def _log_turn(round_number: int, speaker: str, role: str, content: str) -> None:
        """Emit a concise log entry for each discussion turn."""
        logger.info(
            "Discussion round %s | %s (%s): %s",
            round_number,
            speaker,
            role,
            content,
        )

    @staticmethod
    def _format_transcript(transcript: list[DiscussionTurn]) -> str:
        """Format the discussion transcript as plain text."""
        if not transcript:
            return "(no messages yet)"

        return "\n".join(
            f"Round {turn.round_number} - {turn.speaker} ({turn.role}): {turn.content}"
            for turn in transcript
        )
