"""Unit tests for agent custom tools using pytest and pytest-asyncio."""
# pyright: reportAny=false

import pytest
import sys
from pathlib import Path
from pydantic import ValidationError

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from tools import (
    calculator,
    currency_converter,
    database_select,
    loan_calculator,
    current_date,
)


@pytest.mark.asyncio
async def test_calculator_basic_math():
    """Test simple addition and multiplication operations."""
    result = await calculator.ainvoke({"expression": "2 + 2"})
    assert "Result: 4" in result

    result = await calculator.ainvoke({"expression": "10 - 3"})
    assert "Result: 7" in result


@pytest.mark.asyncio
async def test_calculator_precedence():
    """Test standard mathematical operator precedence rules."""
    result = await calculator.ainvoke({"expression": "2 + 2 * 3"})
    assert "Result: 8" in result


@pytest.mark.asyncio
async def test_calculator_invalid():
    """Test behavior with malformed expressions."""
    result = await calculator.ainvoke({"expression": "2 + "})
    assert "Error" in result


@pytest.mark.asyncio
async def test_currency_converter_supported():
    """Test converting between supported currencies."""
    result = await currency_converter.ainvoke(
        {"amount": 100.0, "from_currency": "USD", "to_currency": "EUR"}
    )
    assert "100.00 USD" in result
    assert "EUR" in result


@pytest.mark.asyncio
async def test_currency_converter_unsupported():
    """Test converting with unsupported currencies."""
    result = await currency_converter.ainvoke(
        {"amount": 100.0, "from_currency": "XYZ", "to_currency": "EUR"}
    )
    assert "Error" in result
    assert "not supported" in result


@pytest.mark.asyncio
async def test_loan_calculator_valid():
    """Test valid mortgage loan computation."""
    result = await loan_calculator.ainvoke(
        {"principal": 10000.0, "annual_rate": 5.0, "years": 1}
    )
    assert "Monthly Payment:" in result
    assert "Total Interest:" in result


@pytest.mark.asyncio
async def test_loan_calculator_invalid():
    """Test loan calculations with invalid/negative parameters raises ValidationError."""
    with pytest.raises(ValidationError):
        await loan_calculator.ainvoke(
            {"principal": -1000.0, "annual_rate": 5.0, "years": 5}
        )


@pytest.mark.asyncio
async def test_current_date():
    """Test that current date retrieves structured output formats."""
    result = await current_date.ainvoke({"with_date": True, "with_time": False})
    # Output should match format YYYY-MM-DD
    parts = result.split("-")
    assert len(parts) == 3
    assert len(parts[0]) == 4  # Year
    assert len(parts[1]) == 2  # Month
    assert len(parts[2]) == 2  # Day


@pytest.mark.asyncio
async def test_database_select_runs_select_with_current_sqlalchemy_config(monkeypatch):
    """Test database_select executes a read-only query and returns rows."""

    class FakeResult:
        def keys(self):
            return ["id", "name"]

        def fetchmany(self, limit):
            assert limit == 2
            return [(1, "Ada"), (2, "Grace")]

    class FakeConnection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return False

        def execute(self, statement):
            assert str(statement) == "SELECT id, name FROM dbo.Users"
            return FakeResult()

    class FakeEngine:
        def connect(self):
            return FakeConnection()

    created_urls = []

    def fake_create_engine(url):
        created_urls.append(url)
        return FakeEngine()

    monkeypatch.setenv("DB_HOST", "localhost")
    monkeypatch.setenv("DB_NAME", "appdb")
    monkeypatch.setenv("DB_USERNAME", "dbuser")
    monkeypatch.setenv("DB_PASSWORD", "secret")
    monkeypatch.setenv("DB_DRIVER", "ODBC Driver 17 for SQL Server")
    monkeypatch.setattr("tools.create_engine", fake_create_engine)

    result = await database_select.ainvoke(
        {"query": "SELECT id, name FROM dbo.Users", "limit": 2}
    )

    assert "Rows returned: 2" in result
    assert '"id": 1' in result
    assert '"name": "Grace"' in result
    assert created_urls == [
        "mssql+pyodbc://dbuser:secret@localhost/appdb?driver=ODBC+Driver+17+for+SQL+Server&TrustServerCertificate=yes"
    ]


@pytest.mark.asyncio
async def test_database_select_rejects_write_statement():
    """Test database_select refuses non-SELECT SQL."""
    result = await database_select.ainvoke({"query": "DELETE FROM dbo.Users"})

    assert "Error" in result
    assert "Only read-only SELECT queries are allowed" in result


@pytest.mark.asyncio
async def test_database_select_requires_database_configuration(monkeypatch):
    """Test database_select reports missing DB settings before connecting."""
    for env_name in ["DB_HOST", "DB_NAME", "DB_USERNAME", "DB_PASSWORD"]:
        monkeypatch.delenv(env_name, raising=False)

    result = await database_select.ainvoke({"query": "SELECT 1"})

    assert "Error" in result
    assert "Database configuration is missing" in result
