"""Integration tests for FastAPI endpoints using pytest and mocked agent executor."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

import api
from fastapi.testclient import TestClient
from api import RecipeAndBudgetAnalysis

# Create test client
client = TestClient(api.app)


def test_root_endpoint():
    """Test the root API endpoint returns status 200 and basic documentation info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "LangChain Agent API"
    assert "/docs" in data["docs"]


def test_health_endpoint():
    """Test health check endpoint details."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "agent_loaded" in data


def test_structured_query_mocked():
    """Test POST /structured-query with mocked agent execution."""
    # Store original state to restore after test
    original_loaded = api.AGENT_LOADED
    original_app = api.agent_app

    try:
        # Override globals in api.py
        api.AGENT_LOADED = True
        api.agent_app = MagicMock()

        # Mock the run method to return a standard LangGraph style message dictionary
        mock_msg = MagicMock()
        mock_msg.content = "Mocked recipe recommendation: spinach salad."
        mock_response = {"messages": [mock_msg]}
        api.agent_app.run = AsyncMock(return_value=mock_response)

        payload = {
            "basket": ["spinach", "cucumber"],
            "amount_of_money": 15.0,
            "desire_to_buy": ["olive oil"],
            "query": "What can I cook?",
        }

        response = client.post("/structured-query", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "spinach salad" in data["output"]
        assert "Vegetables in my basket" in data["formatted_prompt"]

    finally:
        # Restore original state
        api.AGENT_LOADED = original_loaded
        api.agent_app = original_app


@patch("services.llm_factory.LLMFactory.create_llm")
def test_structured_output_mocked(mock_create_llm):
    """Test POST /structured-output with mocked structured LLM invocation."""
    # 1. Setup mock LLM and its with_structured_output return value
    mock_llm = MagicMock()
    mock_create_llm.return_value = mock_llm

    mock_structured_llm = MagicMock()
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # 2. Build mock Pydantic response
    mock_response = RecipeAndBudgetAnalysis(
        can_afford_all=True,
        total_estimated_cost=8.5,
        remaining_budget=11.5,
        affordable_items=["olive oil"],
        missing_items=[],
        suggested_recipes=["Fresh Spinach & Cucumber Salad"],
        explanation="Estimated cost matches your available budget.",
    )
    mock_structured_llm.ainvoke = AsyncMock(return_value=mock_response)

    payload = {
        "basket": ["spinach", "cucumber"],
        "amount_of_money": 20.0,
        "desire_to_buy": ["olive oil"],
        "query": "Is it within my budget?",
    }

    response = client.post("/structured-output", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert data["can_afford_all"] is True
    assert data["total_estimated_cost"] == 8.5
    assert "Fresh Spinach & Cucumber Salad" in data["suggested_recipes"]
    assert "available budget" in data["explanation"]
