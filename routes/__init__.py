"""Routes package initialization."""

from . import health_routes
from . import agent_routes
from . import config_routes

__all__ = ["health_routes", "agent_routes", "config_routes"]
