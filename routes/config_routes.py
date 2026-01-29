"""Configuration route handlers."""

import logging
from fastapi import APIRouter, HTTPException
from typing import Optional

from main import AgentApp

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="", tags=["Configuration"])

# Module-level variables to be set by main app
agent_app: Optional[AgentApp] = None
AGENT_LOADED: bool = False


def set_agent_app(app: Optional[AgentApp], loaded: bool) -> None:
    """Set the agent app for configuration access."""
    global agent_app, AGENT_LOADED
    agent_app = app
    AGENT_LOADED = loaded


@router.get("/config")
async def get_config():
    """Get agent configuration details."""
    if not AGENT_LOADED or not agent_app:
        raise HTTPException(status_code=503, detail="Agent not loaded")

    try:
        # Access configurator components
        db_manager = getattr(agent_app, "db_manager", None)
        tools_manager = getattr(agent_app, "tools_manager", None)
        llm = getattr(agent_app, "llm", None)

        return {
            "model": llm.model_name if llm and hasattr(llm, "model_name") else "unknown",
            "temperature": llm.temperature if llm and hasattr(llm, "temperature") else 0.0,
            "tools_count": len(tools_manager.get_tools()) if tools_manager else 0,
            "database_tables": (
                db_manager.include_tables
                if db_manager and hasattr(db_manager, "include_tables")
                else []
            ),
            "sql_tools_enabled": db_manager is not None,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Config error: {str(e)}")
