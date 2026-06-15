"""Unit tests for agent custom tools using pytest and pytest-asyncio."""
# pyright: reportAny=false

import pytest
import sys
from pathlib import Path
from pydantic import ValidationError

# Add parent directory to path to allow importing modules
sys.path.append(str(Path(__file__).parent.parent.resolve()))

from tools import calculator, currency_converter, loan_calculator, current_date


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
