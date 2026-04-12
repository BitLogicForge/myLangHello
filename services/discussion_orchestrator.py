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


class DiscussionOrchestrator:
    """Run a bounded discussion between three persona agents and a moderator."""

    DEFAULT_PERSONAS = (
        AgentPersona(
            name="Titan",
            role="dominant hype strategist",
            instruction=(
                "You are Titan, a hyper-confident alpha-style strategist. You speak with strong conviction, "
                "push bold moves, care about winning attention, and hate timid ideas. Be punchy, persuasive, "
                "and slightly dramatic, but still helpful."
            ),
        ),
        AgentPersona(
            name="Sterling",
            role="business operator",
            instruction=(
                "You are Sterling, a business-minded operator. Think in terms of leverage, delivery risk, "
                "ROI, stakeholder perception, and what can realistically ship fast. Be crisp, practical, "
                "and commercially minded."
            ),
        ),
        AgentPersona(
            name="Riot",
            role="chaotic trend chaser",
            instruction=(
                "You are Riot, an impulsive internet-brained trend chaser. You care about virality, vibes, "
                "memorable moments, and what would make people say 'wait, that is actually cool.' You may "
                "suggest risky or slightly absurd ideas, but keep them usable for a product discussion."
            ),
        ),
        AgentPersona(
            name="Nova",
            role="teenage future scout",
            instruction=(
                "You are Nova, a sharp teenager with strong opinions about what feels outdated, boring, or "
                "actually exciting. You care about novelty, taste, and whether the experience feels fresh. "
                "Be direct, opinionated, and concise."
            ),
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
                content = self._invoke_text(prompt_messages)
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

    def _invoke_text(self, messages: list[BaseMessage]) -> str:
        """Invoke the LLM and normalize the returned text."""
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
