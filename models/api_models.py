"""Pydantic models for API requests and responses."""

from typing import Optional

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


class QueryResponse(BaseModel):
    """Response model for agent queries."""

    output: str = Field(..., description="Agent response")
    session_id: Optional[str] = Field(None, description="Session ID")


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    agent_loaded: bool
    langserve_available: bool
