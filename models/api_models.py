"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field


class MessageHistory(BaseModel):
    """Individual message in conversation history."""

    role: str = Field(..., description="Role of the message sender (human, ai, system)")
    content: str = Field(..., description="Content of the message")


class QueryRequest(BaseModel):
    """Request model for agent queries."""

    question: str = Field(..., description="Question to ask the agent")
    session_id: str | None = Field(None, description="Session ID for conversation tracking")
    user_id: str | None = Field(None, description="User ID for personalization")
    history: list[MessageHistory] | None = Field(
        None,
        description="Conversation history as a list of messages",
    )


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    output: str = Field(..., description="Agent response")
    session_id: str | None = Field(None, description="Session ID")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_loaded: bool
    langserve_available: bool
