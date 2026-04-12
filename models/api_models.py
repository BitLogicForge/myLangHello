"""Pydantic models for API requests and responses."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class MessageHistory(BaseModel):
    """Individual message in conversation history."""

    role: str = Field(..., description="Role of the message sender (human, ai, system)")
    content: str = Field(..., description="Content of the message")


class QueryRequest(BaseModel):
    """Request model for agent queries."""

    question: str = Field(..., description="Question to ask the agent")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    history: Optional[list[MessageHistory]] = Field(
        None,
        description="Conversation history as a list of messages",
    )
    mode: Literal["single", "discussion"] = Field(
        "single",
        description="Execution mode for the request",
    )
    discussion_rounds: int = Field(
        2,
        ge=1,
        le=5,
        description="Number of discussion rounds when mode is discussion",
    )
    include_discussion_transcript: bool = Field(
        True,
        description="Include the agent discussion transcript in the response",
    )


class DiscussionTurn(BaseModel):
    """Single turn inside the multi-agent discussion transcript."""

    round_number: int = Field(..., description="Discussion round number")
    speaker: str = Field(..., description="Agent or moderator name")
    role: str = Field(..., description="Persona role used in the discussion")
    content: str = Field(..., description="Message content for this turn")


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    output: str = Field(..., description="Agent response")
    session_id: Optional[str] = Field(None, description="Session ID")
    mode: Literal["single", "discussion"] = Field(
        "single",
        description="Execution mode used to produce the response",
    )
    transcript: Optional[list[DiscussionTurn]] = Field(
        None,
        description="Discussion transcript when multi-agent discussion mode is used",
    )
    participants: Optional[list[str]] = Field(
        None,
        description="Participants used in discussion mode",
    )


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_loaded: bool
    langserve_available: bool
