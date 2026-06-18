"""Agent tools - all utility functions with LangChain tool decorators."""

import ast
import asyncio
import json
import operator
import random

from datetime import datetime
from pathlib import Path
from typing import Callable, TypedDict

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from utils import is_str_dict, is_list

_ = load_dotenv()


class PortfolioHolding(TypedDict):
    shares: float
    average_cost: float
    total_cost: float


class PortfolioData(TypedDict):
    id: str
    name: str
    initial_capital: float
    current_value: float
    holdings: dict[str, PortfolioHolding]
    created_at: str

# Sandbox folder inside workspace directory
SANDBOX_DIR = Path(__file__).parent.resolve() / "sandbox"


def _safe_path(user_path: str) -> Path:
    """Resolve user path relative to sandbox folder and ensure it doesn't escape."""
    SANDBOX_DIR.mkdir(exist_ok=True)
    clean_path = user_path.lstrip("/\\")
    resolved_path = (SANDBOX_DIR / clean_path).resolve()
    if not resolved_path.is_relative_to(SANDBOX_DIR):
        raise PermissionError("Access denied: path is outside the sandbox.")
    return resolved_path


# ==========================================
# MARK: Calculator
# ==========================================


class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="The mathematical expression to evaluate safely (e.g. '2 + 2 * (3 - 1)'). Supports basic operations: +, -, *, /, **",
    )


@tool(args_schema=CalculatorInput)
async def calculator(expression: str) -> str:
    """Evaluate a math expression safely and return the result as a string.
    Supports basic operations: +, -, *, /, ** (power)
    Use it when you need to perform calculations, but do not use it for anything else.
    Do not execute any code or access files. Just evaluate the math expression and return the result.
    """
    try:
        # Define allowed operations
        allowed_ops: dict[type, Callable[..., object]] = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,  # pyright: ignore[reportUnknownMemberType]
            ast.Pow: operator.pow,  # pyright: ignore[reportUnknownMemberType]
            ast.USub: operator.neg,
        }

        def eval_node(node: ast.AST) -> object:
            if isinstance(node, ast.Num):  # pyright: ignore[reportDeprecated]
                return node.n  # pyright: ignore[reportDeprecated]
            elif isinstance(node, ast.Constant):  # Python 3.8+ uses Constant
                return node.value
            elif isinstance(node, ast.BinOp):  # Binary operation
                if type(node.op) not in allowed_ops:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return allowed_ops[type(node.op)](
                    eval_node(node.left), eval_node(node.right)
                )
            elif isinstance(node, ast.UnaryOp):  # Unary operation (e.g., -5)
                if type(node.op) not in allowed_ops:
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return allowed_ops[type(node.op)](eval_node(node.operand))
            else:
                raise ValueError(f"Unsupported expression type: {type(node).__name__}")

        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree.body)
        return f"Result: {result}"
    except (SyntaxError, ValueError) as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


# ==========================================
# MARK: Weather
# ==========================================


class WeatherInput(BaseModel):
    city: str = Field(
        ..., description="The city name to get weather for (e.g. 'Poznan', 'London')."
    )


@tool(args_schema=WeatherInput)
async def weather(city: str) -> str:
    """Return a fake weather report for the given city."""
    temp_c = random.randint(-10, 35)
    possible_conditions = ["sunny", "cloudy", "rainy", "windy", "snowy"]
    condition = random.choice(possible_conditions)
    return f"The weather in {city} is {condition} and {temp_c}°C."


# ==========================================
# MARK: Read File
# ==========================================


class ReadFileInput(BaseModel):
    path: str = Field(
        ...,
        description="The path of the file to read (relative to the safe sandbox folder).",
    )


@tool(args_schema=ReadFileInput)
async def read_file(path: str) -> str:
    """Read a file from the sandbox directory and return its contents or an error message."""
    try:
        safe_p = _safe_path(path)
        if not safe_p.exists() or not safe_p.is_file():
            return f"Error: file not found: {path}"
        return await asyncio.to_thread(safe_p.read_text, encoding="utf-8")
    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return f"Error reading file: {e}"


# ==========================================
# MARK: Write File
# ==========================================


class WriteFileInput(BaseModel):
    path: str = Field(
        ...,
        description="The path of the file to write to (relative to the safe sandbox folder).",
    )
    content: str = Field(..., description="The text content to write into the file.")


@tool(args_schema=WriteFileInput)
async def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox directory. Return success or error message."""
    try:
        safe_p = _safe_path(path)
        await asyncio.to_thread(safe_p.parent.mkdir, parents=True, exist_ok=True)
        _ = await asyncio.to_thread(safe_p.write_text, content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to sandbox file: {path}"
    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return f"Error writing file: {e}"


# ==========================================
# MARK: Current Date/Time
# ==========================================


class CurrentDateInput(BaseModel):
    with_date: bool = Field(
        default=True, description="Include date (defaults to True)."
    )
    with_time: bool = Field(
        default=False, description="Include time (defaults to False)."
    )


@tool(args_schema=CurrentDateInput)
async def current_date(with_date: bool = True, with_time: bool = False) -> str:
    """Return the current date and/or time as a string.
    Returns date only, time only, or both based on the parameters.
    """
    now = datetime.now()

    if with_date and with_time:
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif with_date:
        return now.strftime("%Y-%m-%d")
    elif with_time:
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d")  # Default to date if both are False


# ==========================================
# MARK: HTTP Get
# ==========================================


class HttpGetInput(BaseModel):
    url: str = Field(..., description="The HTTP/HTTPS URL to perform a GET request on.")


@tool(args_schema=HttpGetInput)
async def http_get(url: str) -> str:
    """Perform an HTTP GET and return a short summary/result."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
        summary = f"Status: {resp.status_code}; Length: {len(resp.content)}"
        try:
            text_preview = resp.text[:1000]
            return summary + "\n" + text_preview
        except Exception:
            return summary
    except Exception as e:
        return f"HTTP GET error: {e}"


# ==========================================
# MARK: Random Joke
# ==========================================


class RandomJokeInput(BaseModel):
    query: str = Field(
        default="",
        description="Optional keyword search query to filter jokes by topic (e.g. 'bug', 'Java').",
    )


@tool(args_schema=RandomJokeInput)
async def random_joke(query: str = "") -> str:
    """Return a small, harmless random joke. Filters by the query keyword if provided.
    Enforces that the last word of the joke is capitalized.
    """
    jokes = [
        "Why do programmers prefer dark mode? Because light attracts bugs.",
        "I told my computer I needed a break, and it said 'No problem — I'll go to sleep.'",
        "There are 10 kinds of people: those who understand binary and those who don't.",
        "Why did the developer go broke? Because he used up all his cache.",
        "Why do Java developers wear glasses? Because they don't see sharp.",
        "Why did the function return early? Because it had a lot on its plate.",
        "Why was the computer cold? It left its Windows open.",
        "What do you call 8 hobbits? A hobbyte.",
        "Why did the programmer quit his job? Because he didn't get arrays.",
    ]

    filtered_jokes = jokes
    if query:
        q = query.lower().strip()
        filtered_jokes = [j for j in jokes if q in j.lower()]

    selected_joke = (
        random.choice(filtered_jokes) if filtered_jokes else random.choice(jokes)
    )

    # Capitalize the last word to satisfy the docstring instruction
    words = selected_joke.split()
    if words:
        # strip final punctuation before capitalizing, or just capitalize the token
        last_word = words[-1]
        # remove common trailing punctuation if any (like .) to capitalize word correctly, then re-append
        punctuation = ""
        if last_word and last_word[-1] in ".?!":
            punctuation = last_word[-1]
            last_word = last_word[:-1]
        words[-1] = last_word.upper() + punctuation

    return " ".join(words)


# ==========================================
# MARK: Joke Format
# ==========================================


class JokeFormatInput(BaseModel):
    joke: str = Field(
        ..., description="The raw joke text to format with decorative borders."
    )


@tool(args_schema=JokeFormatInput)
async def joke_format(joke: str) -> str:
    """Format a joke with decorative borders for better presentation. Do not add any extra text."""
    border = "═" * (len(joke) + 2)
    spacex = " " * (len(joke) + 2)
    return f"""
Best joke for you:
╔═{border}═╗
║ {spacex} ║
║  {joke}  ║
║ {spacex} ║
╚═{border}═╝
"""


# ==========================================
# MARK: Loan Calculator
# ==========================================


class LoanCalculatorInput(BaseModel):
    principal: float = Field(
        ..., gt=0, description="The principal loan amount in USD (must be positive)."
    )
    annual_rate: float = Field(
        ...,
        ge=0,
        description="The annual interest rate as a percentage (e.g., 5.5 for 5.5%).",
    )
    years: int = Field(
        ..., gt=0, description="The term of the loan in years (must be positive)."
    )


@tool(args_schema=LoanCalculatorInput)
async def loan_calculator(principal: float, annual_rate: float, years: int) -> str:
    """Calculate loan payments given principal in USD, annual rate, and term in years."""
    try:
        if principal <= 0 or annual_rate < 0 or years <= 0:
            return (
                "Error: Principal and years must be positive, rate must be non-negative"
            )

        # Convert annual rate to monthly and decimal
        monthly_rate = (annual_rate / 100) / 12
        num_payments = years * 12

        # Calculate monthly payment using amortization formula
        if monthly_rate == 0:
            monthly_payment = principal / num_payments
        else:
            monthly_payment = (
                principal
                * (monthly_rate * (1 + monthly_rate) ** num_payments)
                / ((1 + monthly_rate) ** num_payments - 1)
            )

        total_payment = monthly_payment * num_payments
        total_interest = total_payment - principal

        result = f"""Loan Calculator Results:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Principal Amount:    ${principal:,.2f}
Annual Interest:     {annual_rate}%
Loan Term:          {years} years ({num_payments} months)

Monthly Payment:     ${monthly_payment:,.2f}
Total Payment:       ${total_payment:,.2f}
Total Interest:      ${total_interest:,.2f}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Interest Percentage: {(total_interest / principal) * 100:.2f}% of principal
"""
        return result
    except ValueError:
        return "Error: Invalid number format. Use numbers only (e.g., '200000,5.5,30')"
    except Exception as e:
        return f"Error calculating loan: {e}"


# ==========================================
# MARK: Currency Converter
# ==========================================


class CurrencyConverterInput(BaseModel):
    amount: float = Field(
        ..., gt=0, description="The currency amount to convert (must be positive)."
    )
    from_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="The 3-letter currency code to convert from (e.g., 'USD').",
    )
    to_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="The 3-letter currency code to convert to (e.g., 'EUR').",
    )


@tool(args_schema=CurrencyConverterInput)
async def currency_converter(
    amount: float, from_currency: str, to_currency: str
) -> str:
    """Convert amount from one currency to another using simulated exchange rates."""
    try:
        from_curr = from_currency.strip().upper()
        to_curr = to_currency.strip().upper()

        # Fake exchange rates (relative to USD)
        exchange_rates = {
            "USD": 1.0,
            "EUR": 0.92,
            "GBP": 0.79,
            "JPY": 149.50,
            "CAD": 1.35,
            "AUD": 1.52,
            "CHF": 0.88,
            "CNY": 7.24,
            "INR": 83.12,
            "MXN": 17.15,
            "BRL": 4.98,
            "KRW": 1340.50,
            "SGD": 1.34,
            "HKD": 7.82,
            "SEK": 10.45,
            "NOK": 10.68,
            "DKK": 6.87,
            "ZAR": 18.75,
            "NZD": 1.64,
            "THB": 35.80,
        }

        if from_curr not in exchange_rates:
            available = ", ".join(sorted(exchange_rates.keys()))
            return (
                f"Error: '{from_curr}' not supported. Available currencies: {available}"
            )

        if to_curr not in exchange_rates:
            available = ", ".join(sorted(exchange_rates.keys()))
            return (
                f"Error: '{to_curr}' not supported. Available currencies: {available}"
            )

        # Convert to USD first, then to target currency
        usd_amount = amount / exchange_rates[from_curr]
        converted = usd_amount * exchange_rates[to_curr]

        rate = exchange_rates[to_curr] / exchange_rates[from_curr]

        result = f"""Currency Conversion:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{amount:,.2f} {from_curr}  →  {converted:,.2f} {to_curr}

Exchange Rate: 1 {from_curr} = {rate:.4f} {to_curr}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Note: These are simulated rates for demonstration
"""
        return result
    except ValueError:
        return "Error: Invalid amount. First parameter must be a number"
    except Exception as e:
        return f"Error converting currency: {e}"


# ==========================================
# MARK: City Coordinates
# ==========================================


class CityToCoordinatesInput(BaseModel):
    city: str = Field(
        ...,
        description="The city name to find coordinates for (e.g. 'Paris', 'New York').",
    )


@tool(args_schema=CityToCoordinatesInput)
async def city_to_coordinates(city: str) -> str:
    """Find latitude, longitude, country, and timezone for a given city."""
    try:
        url = f"https://geocoding-api.open-meteo.com/v1/search?name={city.strip()}&count=1"
        async with httpx.AsyncClient() as client:
            resp = await client.get(url, timeout=5.0)
        if resp.status_code != 200:
            return f"Error: Geocoding API returned status code {resp.status_code}"

        data: object = resp.json()  # pyright: ignore[reportAny]
        if not is_str_dict(data):
            return "Error: Invalid response format from Geocoding API"

        results = data.get("results")
        if not is_list(results) or not results:
            return f"Error: City '{city}' not found."

        first_result = results[0]
        if not is_str_dict(first_result):
            return f"Error: Invalid data format for '{city}'"

        name = first_result.get("name")
        country = first_result.get("country", "Unknown")
        lat = first_result.get("latitude")
        lon = first_result.get("longitude")
        timezone = first_result.get("timezone", "Unknown")

        return f"City: {name}, Country: {country}, Latitude: {lat}, Longitude: {lon}, Timezone: {timezone}"
    except Exception as e:
        return f"Error finding coordinates: {e}"


# ==========================================
# MARK: Stock Analysis Tools
# ==========================================

# Type for stock information
class StockInfo(BaseModel):
    name: str
    sector: str
    base_price: float
    volatility: float

# Mock stock database for realistic data generation
MOCK_STOCK_DATA: dict[str, dict[str, str | float]] = {
    "AAPL": {"name": "Apple Inc.", "sector": "Technology", "base_price": 175.50, "volatility": 0.02},
    "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "base_price": 378.25, "volatility": 0.018},
    "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "base_price": 141.75, "volatility": 0.025},
    "AMZN": {"name": "Amazon.com Inc.", "sector": "Consumer Cyclical", "base_price": 178.50, "volatility": 0.022},
    "TSLA": {"name": "Tesla Inc.", "sector": "Automotive", "base_price": 248.75, "volatility": 0.045},
    "META": {"name": "Meta Platforms Inc.", "sector": "Technology", "base_price": 505.25, "volatility": 0.028},
    "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "base_price": 875.50, "volatility": 0.035},
    "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Finance", "base_price": 198.30, "volatility": 0.015},
    "V": {"name": "Visa Inc.", "sector": "Finance", "base_price": 280.75, "volatility": 0.012},
    "JNJ": {"name": "Johnson & Johnson", "sector": "Healthcare", "base_price": 162.40, "volatility": 0.011},
    "WMT": {"name": "Walmart Inc.", "sector": "Consumer Defensive", "base_price": 165.20, "volatility": 0.013},
    "PG": {"name": "Procter & Gamble Co.", "sector": "Consumer Defensive", "base_price": 158.90, "volatility": 0.010},
}

def _get_stock_value(key: str, stock_info: dict[str, str | float]) -> float:
    """Helper to safely extract float values from stock info."""
    value = stock_info[key]
    if isinstance(value, str):
        return float(value)
    return value

def _get_stock_string(key: str, stock_info: dict[str, str | float]) -> str:
    """Helper to safely extract string values from stock info."""
    value = stock_info[key]
    if isinstance(value, str):
        return value
    return str(value)

def _generate_realistic_price(symbol: str, days_ago: int = 0) -> dict[str, float]:
    """Generate realistic stock price with realistic volatility."""
    symbol = symbol.upper()
    if symbol not in MOCK_STOCK_DATA:
        raise ValueError(f"Symbol '{symbol}' not found in stock database")

    stock_info = MOCK_STOCK_DATA[symbol]
    base_price: float = _get_stock_value("base_price", stock_info)
    volatility: float = _get_stock_value("volatility", stock_info)

    # Generate price movement with realistic volatility
    random.seed(hash(f"{symbol}_{datetime.now().date()}_{days_ago}"))

    # Generate price for current day
    day_change: float = random.gauss(0, volatility * base_price)
    current_price: float = max(base_price + day_change, base_price * 0.7)

    # Generate OHLC data
    high_low_range: float = current_price * volatility * 1.5
    high: float = current_price + random.uniform(0, high_low_range)
    low: float = current_price - random.uniform(0, high_low_range)
    open_price: float = current_price + random.uniform(-high_low_range/2, high_low_range/2)

    # Calculate change from previous day
    previous_close: float = current_price * 0.98
    change: float = current_price - previous_close
    change_percent: float = (change / previous_close) * 100 if previous_close > 0 else 0

    return {
        "price": current_price,
        "open": open_price,
        "high": high,
        "low": low,
        "previous_close": previous_close,
        "change": change,
        "change_percent": change_percent,
        "volume": float(random.randint(10_000_000, 150_000_000)),
    }

def RISK_DISCLAIMER():
    """Standard investment disclaimer for all tools."""
    return "\n⚠️ INVESTMENT DISCLAIMER: This analysis is for informational purposes only. Past performance does not guarantee future results. Consult a licensed financial advisor for personalized investment advice."


class StockPriceQueryInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')")
    include_details: bool = Field(default=True, description="Include detailed OHLC data and volume")


@tool(args_schema=StockPriceQueryInput)
async def stock_price_query(symbol: str, include_details: bool = True) -> str:
    """Get current stock price and detailed market data for a given symbol."""
    try:
        symbol = symbol.strip().upper()

        if symbol not in MOCK_STOCK_DATA:
            available = ", ".join(list(MOCK_STOCK_DATA.keys())[:10])
            return f"❌ Stock symbol '{symbol}' not found. Available symbols: {available}, etc."

        stock_info = MOCK_STOCK_DATA[symbol]
        company_name = _get_stock_string("name", stock_info)
        sector = _get_stock_string("sector", stock_info)
        price_data = _generate_realistic_price(symbol)

        result = f"""📊 {company_name} ({symbol})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Current Price:      ${price_data['price']:.2f}
📈 Change:            ${price_data['change']:+.2f} ({price_data['change_percent']:+.2f}%)
🏢 Sector:            {sector}
"""

        if include_details:
            result += f"""
📊 Detailed Data:
   Open:           ${price_data['open']:.2f}
   High:           ${price_data['high']:.2f}
   Low:            ${price_data['low']:.2f}
   Previous Close: ${price_data['previous_close']:.2f}
   Volume:         {price_data['volume']:,}

💡 Market Status: {'🟢 Market Open' if random.choice([True, False]) else '🔴 Market Closed'}
"""

        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += RISK_DISCLAIMER()

        return result

    except Exception as e:
        return f"❌ Error retrieving stock price: {str(e)}"


class FinancialMetricsInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol for financial analysis")
    metrics: list[str] = Field(
        default=["all"],
        description="List of metrics: 'pe_ratio', 'market_cap', 'dividend_yield', 'eps', 'beta', 'debt_equity', 'profit_margin', 'all'"
    )


@tool(args_schema=FinancialMetricsInput)
async def financial_metrics_calculator(symbol: str, metrics: list[str] | None = None) -> str:
    """Calculate comprehensive financial metrics and ratios for stock analysis."""
    try:
        if metrics is None:
            metrics = ["all"]

        symbol = symbol.strip().upper()

        if symbol not in MOCK_STOCK_DATA:
            return f"❌ Stock symbol '{symbol}' not found in database."

        stock_info = MOCK_STOCK_DATA[symbol]
        company_name = _get_stock_string("name", stock_info)
        _ = _get_stock_string("sector", stock_info)  # Sector available but not used in current implementation
        price_data = _generate_realistic_price(symbol)

        # Generate realistic financial metrics
        market_cap = random.randint(50_000_000_000, 3_000_000_000_000)
        pe_ratio = random.uniform(15, 45)
        eps = price_data['price'] / pe_ratio
        dividend_yield = random.uniform(0, 3.5)
        beta = random.uniform(0.7, 1.8)
        debt_to_equity = random.uniform(0.1, 1.5)
        profit_margin = random.uniform(8, 28)
        roe = random.uniform(12, 35)
        current_ratio = random.uniform(1.2, 3.5)

        result = f"""📈 Financial Metrics Analysis: {company_name} ({symbol})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Valuation Metrics:
   P/E Ratio:          {pe_ratio:.2f}
   Earnings Per Share: ${eps:.2f}
   Market Cap:         ${market_cap/1_000_000_000:.1f}B
   Price/Book:         {random.uniform(2.5, 15):.2f}

💵 Dividend Information:
   Dividend Yield:     {dividend_yield:.2f}%
   Annual Dividend:    ${(price_data['price'] * dividend_yield / 100):.2f}

📊 Risk & Return Metrics:
   Beta:               {beta:.2f} ({'High volatility' if beta > 1.3 else 'Moderate risk' if beta > 0.9 else 'Low risk'})
   52W Range:          ${price_data['price'] * 0.75:.2f} - ${price_data['price'] * 1.35:.2f}

🏦 Financial Health:
   Debt/Equity:        {debt_to_equity:.2f}
   Profit Margin:      {profit_margin:.2f}%
   Return on Equity:   {roe:.2f}%
   Current Ratio:      {current_ratio:.2f}

📈 Growth Estimates:
   Revenue Growth:     {random.uniform(-2, 25):.1f}% YoY
   EPS Growth:         {random.uniform(-5, 30):.1f}% YoY
"""

        # Investment quality score
        quality_score = 0
        if pe_ratio < 25: quality_score += 20
        if debt_to_equity < 0.8: quality_score += 25
        if profit_margin > 15: quality_score += 25
        if roe > 18: quality_score += 15
        if beta < 1.2: quality_score += 15

        result += f"\n🎯 Investment Quality Score: {quality_score}/100"
        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += RISK_DISCLAIMER()

        return result

    except Exception as e:
        return f"❌ Error calculating financial metrics: {str(e)}"


class PortfolioCreateInput(BaseModel):
    portfolio_name: str = Field(..., description="Name for the portfolio")
    initial_capital: float = Field(..., gt=0, description="Initial investment amount")


@tool(args_schema=PortfolioCreateInput)
async def portfolio_create(portfolio_name: str, initial_capital: float) -> str:
    """Create a new investment portfolio with specified initial capital."""
    try:
        portfolio_id = f"PORT_{random.randint(1000, 9999)}"

        # Store portfolio in memory (simplified - in production use database)
        portfolio_data = {
            "id": portfolio_id,
            "name": portfolio_name,
            "initial_capital": initial_capital,
            "current_value": initial_capital,
            "holdings": {},
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # For demo purposes, store in a file
        SANDBOX_DIR.mkdir(exist_ok=True)
        portfolio_file = SANDBOX_DIR / f"{portfolio_id}.json"
        _ = await asyncio.to_thread(portfolio_file.write_text, json.dumps(portfolio_data))

        created_at: str = str(portfolio_data.get("created_at", "")) or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        result = f"""✅ Portfolio Created Successfully!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📁 Portfolio ID:      {portfolio_id}
🏷️  Name:             {portfolio_name}
💰 Initial Capital:    ${initial_capital:,.2f}
📅 Created:           {created_at}

💡 Next Steps:
   • Add stocks using: portfolio_add_stock
   • View performance: portfolio_analyze
   • Get recommendations: portfolio_optimize

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💾 Portfolio saved as: {portfolio_id}.json
{RISK_DISCLAIMER()}"""

        return result

    except Exception as e:
        return f"❌ Error creating portfolio: {str(e)}"


class PortfolioAddStockInput(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio ID (e.g., 'PORT_1234')")
    symbol: str = Field(..., description="Stock symbol to add")
    shares: float = Field(..., gt=0, description="Number of shares to add")
    buy_price: float = Field(..., gt=0, description="Purchase price per share")


@tool(args_schema=PortfolioAddStockInput)
async def portfolio_add_stock(portfolio_id: str, symbol: str, shares: float, buy_price: float) -> str:
    """Add a stock position to existing portfolio."""
    try:
        symbol = symbol.strip().upper()

        if symbol not in MOCK_STOCK_DATA:
            return f"❌ Stock symbol '{symbol}' not found. Available symbols: {list(MOCK_STOCK_DATA.keys())[:5]} etc."

        # Load portfolio
        portfolio_file = SANDBOX_DIR / f"{portfolio_id}.json"
        if not portfolio_file.exists():
            return f"❌ Portfolio '{portfolio_id}' not found. Create one first with portfolio_create."

        raw_data = await asyncio.to_thread(portfolio_file.read_text)
        portfolio_data: PortfolioData = json.loads(raw_data)

        # Ensure portfolio data is properly typed
        if not isinstance(portfolio_data, dict):
            return f"❌ Invalid portfolio data format"

        # Add stock to holdings with proper typing
        holdings_dict: dict[str, PortfolioHolding] = portfolio_data["holdings"]

        if symbol not in holdings_dict:
            new_holding: PortfolioHolding = {
                "shares": shares,
                "average_cost": buy_price,
                "total_cost": shares * buy_price
            }
            holdings_dict[symbol] = new_holding
        else:
            existing: PortfolioHolding = holdings_dict[symbol]
            existing_shares = float(existing["shares"])
            existing_total_cost = float(existing["total_cost"])

            total_shares = existing_shares + shares
            total_cost = existing_total_cost + (shares * buy_price)

            updated_holding: PortfolioHolding = {
                "shares": total_shares,
                "average_cost": total_cost / total_shares,
                "total_cost": total_cost
            }
            holdings_dict[symbol] = updated_holding

        # Update portfolio data
        portfolio_data["holdings"] = holdings_dict

        # Save updated portfolio
        _ = await asyncio.to_thread(portfolio_file.write_text, json.dumps(portfolio_data))

        holding = holdings_dict[symbol]
        current_value = float(holding["shares"]) * _generate_realistic_price(symbol)["price"]
        total_cost_hold = float(holding["total_cost"])
        profit_loss = current_value - total_cost_hold
        profit_loss_percent = (profit_loss / total_cost_hold) * 100

        result = f"""✅ Stock Added to Portfolio: {portfolio_id}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 Stock Added:        {symbol} - {MOCK_STOCK_DATA[symbol]['name']}
📈 Shares Added:      {shares:.2f}
💵 Buy Price:          ${buy_price:.2f}
💰 Total Cost:         ${shares * buy_price:,.2f}

📁 Updated Position:
   Total Shares:       {portfolio_data['holdings'][symbol]['shares']:.2f}
   Average Cost:       ${portfolio_data['holdings'][symbol]['average_cost']:.2f}
   Current Value:      ${current_value:,.2f}
   P/L:                ${profit_loss:+,.2f} ({profit_loss_percent:+.2f}%)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💾 Portfolio updated: {portfolio_id}.json
{RISK_DISCLAIMER()}"""

        return result

    except Exception as e:
        return f"❌ Error adding stock to portfolio: {str(e)}"


class PortfolioAnalyzeInput(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio ID to analyze")
    include_recommendations: bool = Field(default=True, description="Include optimization recommendations")


@tool(args_schema=PortfolioAnalyzeInput)
async def portfolio_analyze(portfolio_id: str, include_recommendations: bool = True) -> str:
    """Analyze portfolio performance, risk, and provide detailed insights."""
    try:
        # Load portfolio
        portfolio_file = SANDBOX_DIR / f"{portfolio_id}.json"
        if not portfolio_file.exists():
            return f"❌ Portfolio '{portfolio_id}' not found."

        raw_data = await asyncio.to_thread(portfolio_file.read_text)
        portfolio_data: PortfolioData = json.loads(raw_data)

        # Ensure portfolio data is properly typed
        if not isinstance(portfolio_data, dict):
            return f"❌ Invalid portfolio data format"

        if not portfolio_data["holdings"]:
            return f"📊 Portfolio '{portfolio_id}' is empty. Add stocks using portfolio_add_stock."

        # Calculate portfolio metrics with proper typing
        holdings_dict: dict[str, PortfolioHolding] = portfolio_data["holdings"]
        total_cost = sum(float(h["total_cost"]) for h in holdings_dict.values())
        current_value: float = 0.0
        sector_breakdown: dict[str, float] = {}
        positions: list[dict[str, str | float]] = []

        for symbol, holding in holdings_dict.items():
            stock_info = MOCK_STOCK_DATA[symbol]
            price_data = _generate_realistic_price(symbol)
            current_price = price_data["price"]

            shares_float = float(holding["shares"])
            avg_cost_float = float(holding["average_cost"])
            total_cost_float = float(holding["total_cost"])

            position_value = shares_float * current_price
            current_value += position_value

            profit_loss = position_value - total_cost_float
            profit_loss_percent = (profit_loss / total_cost_float) * 100

            sector = _get_stock_string("sector", stock_info)
            sector_breakdown[sector] = sector_breakdown.get(sector, 0.0) + position_value

            positions.append({
                "symbol": str(symbol),
                "name": str(_get_stock_string("name", stock_info)),
                "shares": shares_float,
                "avg_cost": avg_cost_float,
                "current_price": current_price,
                "value": position_value,
                "p_l": profit_loss,
                "p_l_percent": profit_loss_percent,
                "sector": sector
            })

        total_return = current_value - total_cost
        total_return_percent = (total_return / total_cost) * 100 if total_cost > 0 else 0

        # Calculate portfolio metrics
        portfolio_beta = sum(float(_get_stock_value("base_price", MOCK_STOCK_DATA[str(pos["symbol"])])) * 0.001 for pos in positions) / len(positions)
        portfolio_volatility = sum(float(_get_stock_value("volatility", MOCK_STOCK_DATA[str(pos["symbol"])])) for pos in positions) / len(positions)

        # Risk assessment
        risk_level = "Low" if portfolio_volatility < 0.02 else "Moderate" if portfolio_volatility < 0.03 else "High"

        result = f"""📊 Portfolio Analysis: {portfolio_data['name']} ({portfolio_id})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Portfolio Summary:
   Initial Capital:    ${portfolio_data['initial_capital']:,.2f}
   Total Invested:     ${total_cost:,.2f}
   Current Value:      ${current_value:,.2f}
   Total Return:       ${total_return:+,.2f} ({total_return_percent:+.2f}%)

🎯 Performance Metrics:
   Portfolio Beta:      {portfolio_beta:.2f}
   Volatility:         {portfolio_volatility:.3f} ({risk_level} Risk)
   Sharpe Ratio:       {random.uniform(0.5, 2.5):.2f}
   Max Drawdown:       {random.uniform(5, 25):.1f}%

📁 Sector Allocation:
"""

        # Sector breakdown
        # Sector breakdown
        sector_items = [(sector, float(value)) for sector, value in sector_breakdown.items()]
        for sector, value in sorted(sector_items, key=lambda item: item[1], reverse=True):
            allocation = (value / current_value) * 100 if current_value > 0 else 0
            result += f"   {sector}:        ${value:,.2f} ({allocation:.1f}%)\n"

        result += "\n📋 Position Details:\n"
        # Calculate portfolio metrics with proper typing
        positions_list: list[tuple[dict[str, str | float], float]] = []
        for pos in positions:
            pos_value = float(pos["value"])
            positions_list.append((pos, pos_value))

        sorted_position_items = sorted(positions_list, key=lambda item: item[1], reverse=True)

        for pos, pos_value in sorted_position_items:
            result += f"""
   {str(pos['symbol'])} - {str(pos['name'])}
   Shares: {float(pos['shares']):.2f} | Avg Cost: ${float(pos['avg_cost']):.2f} | Current: ${float(pos['current_price']):.2f}
   Value: ${float(pos_value):,.2f} | P/L: ${float(pos['p_l']):+,.2f} ({float(pos['p_l_percent']):+.2f}%)
"""

        if include_recommendations:
            # Generate recommendations based on portfolio analysis
            recommendations: list[str] = []

            if len(positions) < 5:
                recommendations.append("🔸 Consider diversifying - add more positions from different sectors")
            if portfolio_volatility > 0.03:
                recommendations.append("🔸 High volatility detected - consider adding defensive stocks")
            if total_return_percent < 0:
                recommendations.append("🔸 Portfolio showing losses - review underperforming positions")

            # Check for positions with significant losses with proper typing
            position_pl_percents: list[float] = []
            for pos in positions:
                pl_percent = pos["p_l_percent"]
                position_pl_percents.append(float(pl_percent))

            has_big_losses = any(pl < -15 for pl in position_pl_percents)
            if has_big_losses:
                recommendations.append("🔸 Some positions showing significant losses - consider stop-loss strategies")

            if portfolio_beta > 1.2:
                recommendations.append("🔸 High beta portfolio - consider adding lower volatility stocks")

            result += "\n💡 Recommendations:\n"
            for rec in recommendations:
                result += f"   {rec}\n"

        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += RISK_DISCLAIMER()

        return result

    except Exception as e:
        return f"❌ Error analyzing portfolio: {str(e)}"


class TechnicalAnalysisInput(BaseModel):
    symbol: str = Field(..., description="Stock symbol for technical analysis")
    indicators: list[str] = Field(
        default=["sma", "rsi", "macd"],
        description="Technical indicators: 'sma', 'ema', 'rsi', 'macd', 'bollinger', 'all'"
    )


@tool(args_schema=TechnicalAnalysisInput)
async def technical_analysis(symbol: str, indicators: list[str] | None = None) -> str:
    """Perform technical analysis with popular trading indicators."""
    try:
        if indicators is None:
            indicators = ["sma", "rsi", "macd"]

        symbol = symbol.strip().upper()

        if symbol not in MOCK_STOCK_DATA:
            return f"❌ Stock symbol '{symbol}' not found."

        stock_info = MOCK_STOCK_DATA[symbol]
        company_name = _get_stock_string("name", stock_info)
        volatility = _get_stock_value("volatility", stock_info)
        price_data = _generate_realistic_price(symbol)
        current_price = price_data["price"]

        # Generate realistic historical prices for calculations
        random.seed(hash(symbol))
        historical_prices = [current_price * (1 + random.gauss(0, volatility)) for _ in range(50)]
        historical_prices.reverse()  # Most recent last

        result = f"""📈 Technical Analysis: {company_name} ({symbol})
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

💰 Current Price: ${current_price:.2f}
"""

        # Simple Moving Averages (SMA)
        if "sma" in indicators or "all" in indicators:
            sma_20 = sum(historical_prices[-20:]) / 20
            sma_50 = sum(historical_prices[-50:]) / 50

            sma_signal = "🟢 BULLISH" if current_price > sma_20 > sma_50 else "🔴 BEARISH" if current_price < sma_20 < sma_50 else "⚪ NEUTRAL"

            result += f"""
📊 Moving Averages:
   SMA 20: ${sma_20:.2f} ({'▲' if current_price > sma_20 else '▼'} {abs((current_price - sma_20) / sma_20 * 100):.2f}%)
   SMA 50: ${sma_50:.2f} ({'▲' if current_price > sma_50 else '▼'} {abs((current_price - sma_50) / sma_50 * 100):.2f}%)
   Signal: {sma_signal}
"""

        # RSI (Relative Strength Index)
        if "rsi" in indicators or "all" in indicators:
            # Simplified RSI calculation
            gains = [max(historical_prices[i] - historical_prices[i-1], 0) for i in range(1, len(historical_prices))]
            losses = [max(historical_prices[i-1] - historical_prices[i], 0) for i in range(1, len(historical_prices))]

            avg_gain = sum(gains[-14:]) / 14
            avg_loss = sum(losses[-14:]) / 14

            rs = avg_gain / avg_loss if avg_loss != 0 else 1
            rsi = 100 - (100 / (1 + rs))

            rsi_signal = "🔴 OVERBOUGHT" if rsi > 70 else "🟢 OVERSOLD" if rsi < 30 else "⚪ NEUTRAL"

            result += f"""
🎯 RSI (14): {rsi:.1f}
   Status: {rsi_signal}
   (70+ = Overbought, 30- = Oversold)
"""

        # MACD (Moving Average Convergence Divergence)
        if "macd" in indicators or "all" in indicators:
            ema_12 = sum(historical_prices[-12:]) / 12  # Simplified
            ema_26 = sum(historical_prices[-26:]) / 26
            macd_line = ema_12 - ema_26
            signal_line = macd_line * 0.9  # Simplified
            histogram = macd_line - signal_line

            macd_signal = "🟢 BULLISH" if macd_line > signal_line else "🔴 BEARISH"

            result += f"""
📊 MACD:
   MACD Line:    {macd_line:.2f}
   Signal Line:  {signal_line:.2f}
   Histogram:    {histogram:.2f}
   Signal:       {macd_signal}
"""

        # Bollinger Bands
        if "bollinger" in indicators or "all" in indicators:
            sma_20 = sum(historical_prices[-20:]) / 20
            std_dev: float = (sum((p - sma_20) ** 2 for p in historical_prices[-20:]) / 20) ** 0.5

            upper_band: float = sma_20 + (2 * std_dev)
            lower_band: float = sma_20 - (2 * std_dev)

            bb_position = (current_price - lower_band) / (upper_band - lower_band) * 100 if upper_band != lower_band else 50
            bb_signal = "🔴 Near Upper Band" if bb_position > 80 else "🟢 Near Lower Band" if bb_position < 20 else "⚪ Middle Range"

            result += f"""
📊 Bollinger Bands (20, 2):
   Upper Band:   ${upper_band:.2f}
   Middle Band:  ${sma_20:.2f}
   Lower Band:   ${lower_band:.2f}
   Position:     {bb_position:.1f}% ({bb_signal})
"""

        # Overall technical rating
        technical_score = random.randint(45, 85)
        rating = "Strong Buy" if technical_score > 75 else "Buy" if technical_score > 65 else "Hold" if technical_score > 50 else "Sell" if technical_score > 35 else "Strong Sell"

        result += f"""
🎯 Technical Rating: {technical_score}/100 - {rating}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{RISK_DISCLAIMER()}"""

        return result

    except Exception as e:
        return f"❌ Error in technical analysis: {str(e)}"


class RiskAssessmentInput(BaseModel):
    portfolio_id: str | None = Field(default=None, description="Optional portfolio ID for risk assessment")
    symbol: str | None = Field(default=None, description="Optional single stock symbol for risk analysis")


@tool(args_schema=RiskAssessmentInput)
async def risk_assessment(portfolio_id: str | None = None, symbol: str | None = None) -> str:
    """Comprehensive risk assessment for portfolios or individual stocks."""
    try:
        result = "🎯 Risk Assessment Report\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        if symbol and not portfolio_id:
            # Single stock risk analysis
            symbol = symbol.strip().upper()

            if symbol not in MOCK_STOCK_DATA:
                return f"❌ Stock symbol '{symbol}' not found."

            stock_info = MOCK_STOCK_DATA[symbol]
            company_name = _get_stock_string("name", stock_info)
            sector = _get_stock_string("sector", stock_info)
            volatility = _get_stock_value("volatility", stock_info)
            price_data = _generate_realistic_price(symbol)

            # Risk metrics calculations
            beta = random.uniform(0.7, 1.8)
            var_95 = price_data["price"] * volatility * 1.65  # Value at Risk
            max_drawdown = random.uniform(15, 45)
            sharpe_ratio = random.uniform(0.3, 2.8)

            # Risk category
            risk_score = volatility * 100 + (beta * 10)
            risk_category = "Low Risk" if risk_score < 3 else "Moderate Risk" if risk_score < 5 else "High Risk"

            result += f"📊 Single Stock Analysis: {company_name} ({symbol})\n\n"
            result += f"""🎯 Risk Category: {risk_category} (Score: {risk_score:.1f}/10)

📈 Volatility Risk:
   Daily Volatility:    {volatility * 100:.2f}%
   Annual Volatility:  {volatility * 100 * 16:.2f}%
   Beta:               {beta:.2f} ({'High' if beta > 1.2 else 'Moderate' if beta > 0.8 else 'Low'} market correlation)

💴 Downside Risk:
   VaR (95%):          -${var_95:.2f} per share (potential 1-day loss)
   Max Drawdown:        {max_drawdown:.1f}% (historical maximum decline)
   Stop Loss Price:     ${price_data['price'] * 0.92:.2f} (recommended 8% below current)

🎪 Risk-Adjusted Returns:
   Sharpe Ratio:        {sharpe_ratio:.2f}
   Sortino Ratio:       {sharpe_ratio * 0.85:.2f}
   Risk-Free Rate:      4.5% (current 10-year Treasury)

🔍 Risk Factors:
   Sector Risk:         {sector} ({'High' if sector in ['Technology', 'Automotive'] else 'Moderate' if sector in ['Finance', 'Consumer Cyclical'] else 'Low'} volatility sector)
   Market Cap Risk:     {'Large Cap' if price_data['price'] > 200 else 'Mid Cap' if price_data['price'] > 100 else 'Small Cap'}
   Concentration Risk:  High (single stock exposure)

🛡️ Risk Mitigation:
   • Consider position sizing: limit to 5-10% of portfolio
   • Use stop-loss orders to limit downside
   • Consider hedging with options or diversification
   • Monitor earnings and sector trends
"""

        elif portfolio_id and not symbol:
            # Portfolio risk analysis
            portfolio_file = SANDBOX_DIR / f"{portfolio_id}.json"
            if not portfolio_file.exists():
                return f"❌ Portfolio '{portfolio_id}' not found."

            raw_data = await asyncio.to_thread(portfolio_file.read_text)
            portfolio_data: PortfolioData = json.loads(raw_data)

            holdings_dict: dict[str, PortfolioHolding] = portfolio_data["holdings"]

            if not holdings_dict:
                return f"📊 Portfolio is empty - no risk analysis available."

            # Calculate portfolio risk metrics with proper typing
            symbols = list(holdings_dict.keys())
            portfolio_volatility = sum(float(_get_stock_value("volatility", MOCK_STOCK_DATA[s])) for s in symbols) / len(symbols)
            portfolio_beta = sum(random.uniform(0.7, 1.8) for _ in symbols) / len(symbols)

            # Concentration risk with proper typing
            total_value = 0.0
            for s in symbols:
                holding = holdings_dict[s]
                shares = float(holding["shares"])
                price_data = _generate_realistic_price(s)
                total_value += shares * price_data["price"]

            concentrations: list[tuple[str, float]] = []
            for s in symbols:
                holding = holdings_dict[s]
                shares = float(holding["shares"])
                price_data = _generate_realistic_price(s)
                position_value = shares * price_data["price"]
                concentration = (position_value / total_value * 100) if total_value > 0 else 0
                concentrations.append((s, concentration))

            concentrations.sort(key=lambda x: x[1], reverse=True)

            max_concentration = concentrations[0][1] if concentrations else 0
            concentration_risk = "High" if max_concentration > 25 else "Moderate" if max_concentration > 15 else "Low"

            # Sector diversification
            sectors = [_get_stock_string("sector", MOCK_STOCK_DATA[s]) for s in symbols]
            sector_diversification = len(set(sectors))

            result += f"📊 Portfolio Risk Analysis: {portfolio_data['name']} ({portfolio_id})\n\n"
            result += f"""🎯 Overall Risk Level: {'High' if portfolio_volatility > 0.025 else 'Moderate' if portfolio_volatility > 0.018 else 'Low'} Risk Portfolio

📈 Portfolio Volatility:
   Daily Volatility:    {portfolio_volatility * 100:.2f}%
   Annual Volatility:  {portfolio_volatility * 100 * 16:.2f}%
   Portfolio Beta:      {portfolio_beta:.2f}

💼 Concentration Risk: {concentration_risk}
   Largest Position:    {concentrations[0][0]} ({concentrations[0][1]:.1f}% of portfolio)
   Top 3 Holdings:      {sum(c[1] for c in concentrations[:3]):.1f}% of portfolio
   Number of Positions: {len(symbols)}

🏢 Sector Diversification:
   Different Sectors:   {sector_diversification}
   Sectors Covered:     {', '.join(set(sectors))}
   Recommendation:      {'Add more sectors for diversification' if sector_diversification < 4 else 'Good sector diversification'}

💴 Downside Risk:
   Portfolio VaR (95%): -${total_value * portfolio_volatility * 1.65:,.2f} (potential 1-day loss)
   Max Drawdown Risk:    {portfolio_volatility * 100 * random.uniform(3, 6):.1f}%
   Correlation Risk:     {'High' if sector_diversification < 3 else 'Moderate' if sector_diversification < 5 else 'Low'}

🛡️ Risk Mitigation Strategies:
   • {'Reduce concentration in ' + concentrations[0][0] if max_concentration > 25 else 'Maintain current position sizing'}
   • {'Consider adding defensive sectors (Utilities, Consumer Staples)' if 'Technology' in sectors and sector_diversification < 4 else 'Good sector balance'}
   • Use stop-loss orders on individual positions
   • Consider rebalancing if drift exceeds 5%
   • Regular portfolio reviews recommended
"""

        else:
            return "❌ Please provide either a portfolio_id or symbol for risk assessment."

        result += "\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += RISK_DISCLAIMER()

        return result

    except Exception as e:
        return f"❌ Error in risk assessment: {str(e)}"


class MarketDataInput(BaseModel):
    data_type: str = Field(..., description="Type of market data: 'indices', 'sectors', 'movers', 'overview'")
    timeframe: str = Field(default="today", description="Time period: 'today', 'week', 'month'")


@tool(args_schema=MarketDataInput)
async def market_data(data_type: str = "overview", timeframe: str = "today") -> str:
    """Get comprehensive market data including indices, sector performance, and market movers."""
    try:
        result = f"📊 Market Data: {data_type.upper()} ({timeframe})\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"

        if data_type == "indices":
            # Major market indices
            indices = {
                "S&P 500": {"value": 5234.18, "change": 45.23, "change_percent": 0.87},
                "DOW JONES": {"value": 39127.80, "change": -125.43, "change_percent": -0.32},
                "NASDAQ": {"value": 16439.22, "change": 183.02, "change_percent": 1.12},
                "RUSSELL 2000": {"value": 2089.45, "change": -12.67, "change_percent": -0.60},
            }

            result += "🏛️ Major Market Indices:\n\n"
            for index_name, data in indices.items():
                change_symbol = "▲" if data["change"] > 0 else "▼"
                result += f"   {index_name:12s}: {data['value']:,.2f}  {change_symbol} {data['change']:+,.2f} ({data['change_percent']:+.2f}%)\n"

        elif data_type == "sectors":
            # Sector performance
            sectors = {
                "Technology": {"change": 1.45, "status": "Outperforming"},
                "Healthcare": {"change": 0.82, "status": "Performing"},
                "Finance": {"change": -0.34, "status": "Underperforming"},
                "Consumer Cyclical": {"change": 0.67, "status": "Performing"},
                "Energy": {"change": -1.23, "status": "Lagging"},
                "Utilities": {"change": 0.23, "status": "Defensive"},
                "Real Estate": {"change": -0.56, "status": "Underperforming"},
                "Materials": {"change": 0.45, "status": "Performing"},
            }

            result += "🏢 Sector Performance:\n\n"
            for sector, data in sorted(sectors.items(), key=lambda x: float(x[1]["change"]), reverse=True):
                change_value = float(data["change"])
                change_symbol = "🟢" if change_value > 0 else "🔴"
                status_symbol = "🚀" if data["status"] == "Outperforming" else "📊" if data["status"] == "Performing" else "🛡️" if data["status"] == "Defensive" else "⚠️"
                result += f"   {change_symbol} {sector:20s}: {change_value:+.2f}%  {status_symbol} {data['status']}\n"

        elif data_type == "movers":
            # Top gainers and losers
            gainers = [
                {"symbol": "NVDA", "name": "NVIDIA", "change": 5.67, "price": 875.50},
                {"symbol": "TSLA", "name": "Tesla", "change": 4.23, "price": 248.75},
                {"symbol": "META", "name": "Meta", "change": 3.45, "price": 505.25},
            ]

            losers = [
                {"symbol": "JPM", "name": "JPMorgan", "change": -2.34, "price": 198.30},
                {"symbol": "WMT", "name": "Walmart", "change": -1.89, "price": 165.20},
                {"symbol": "PG", "name": "Procter & Gamble", "change": -1.45, "price": 158.90},
            ]

            result += "🚀 Top Gainers:\n\n"
            for stock in gainers:
                result += f"   🟢 {stock['symbol']} ({stock['name']}): ${stock['price']:.2f}  ▲ +{stock['change']:.2f}%\n"

            result += "\n📉 Top Losers:\n\n"
            for stock in losers:
                result += f"   🔴 {stock['symbol']} ({stock['name']}): ${stock['price']:.2f}  ▼ {stock['change']:.2f}%\n"

        elif data_type == "overview":
            # Market overview
            result += """🌍 Market Overview:

🏛️ Market Status:        🟢 Open (Regular Hours)
📊 Market Sentiment:     ⚪ Neutral to Bullish
📈 Volatility Index (VIX): 14.25 (Moderate)
💵 10-Year Treasury:     4.52%
🛢️ Crude Oil (WTI):      $78.45
💰 Gold (COMEX):         $2,345.80

🎯 Key Levels:
   S&P 500 Support:      5,180 | Resistance: 5,280
   DOW Support:          39,000 | Resistance: 39,500

📊 Trading Volume:       Above average
🏦 Market Breadth:       62% Advancing vs 38% Declining

💡 Market Focus:
   • Fed Rate Policy:        Next meeting in 3 weeks
   • Earnings Season:        Active (Tech sector reporting)
   • Economic Calendar:      CPI data released tomorrow
   • Geopolitical Events:    Monitoring Middle East situation

🎪 Market Movers:
   • Technology sector leading gains
   • Energy sector under pressure
   • Financials mixed on rate concerns"""

        result += "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += RISK_DISCLAIMER()

        return result

    except Exception as e:
        return f"❌ Error retrieving market data: {str(e)}"


class StockComparisonInput(BaseModel):
    symbols: list[str] = Field(description="Stock symbols to compare (2-5 symbols)", min_length=2, max_length=5)
    comparison_type: str = Field(default="overview", description="Type: 'overview', 'financial', 'technical', 'risk'")


@tool(args_schema=StockComparisonInput)
async def stock_comparison(symbols: list[str], comparison_type: str = "overview") -> str:
    """Compare multiple stocks across different analysis dimensions."""
    try:
        symbols = [s.strip().upper() for s in symbols]

        # Validate symbols
        valid_symbols = [s for s in symbols if s in MOCK_STOCK_DATA]
        if not valid_symbols:
            return f"❌ None of the symbols found. Available: {list(MOCK_STOCK_DATA.keys())[:5]} etc."

        result = f"📊 Stock Comparison: {comparison_type.upper()}\n"
        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"

        if comparison_type == "overview":
            result += f"{'Symbol':<8} {'Name':<25} {'Price':<10} {'Change%':<10} {'Sector':<20}\n"
            result += "─" * 73 + "\n"

            for symbol in valid_symbols:
                stock_info = MOCK_STOCK_DATA[symbol]
                price_data = _generate_realistic_price(symbol)
                change_symbol = "▲" if price_data["change_percent"] > 0 else "▼"

                result += f"{symbol:<8} {stock_info['name']:<25} ${price_data['price']:<9.2f} {change_symbol} {price_data['change_percent']:+<8.2f}% {stock_info['sector']:<20}\n"

        elif comparison_type == "financial":
            result += f"{'Symbol':<8} {'P/E':<8} {'Market Cap':<12} {'Div%':<8} {'Beta':<8} {'Debt/Eq':<10}\n"
            result += "─" * 54 + "\n"

            for symbol in valid_symbols:
                pe_ratio = random.uniform(15, 45)
                market_cap = random.randint(50, 3000)
                dividend_yield = random.uniform(0, 3.5)
                beta = random.uniform(0.7, 1.8)
                debt_equity = random.uniform(0.1, 1.5)

                result += f"{symbol:<8} {pe_ratio:<8.2f} ${market_cap:<11.0f}B {dividend_yield:<8.2f}% {beta:<8.2f} {debt_equity:<10.2f}\n"

        elif comparison_type == "technical":
            result += f"{'Symbol':<8} {'SMA20':<10} {'RSI':<8} {'MACD':<10} {'Signal':<15}\n"
            result += "─" * 51 + "\n"

            for symbol in valid_symbols:
                price_data = _generate_realistic_price(symbol)
                sma_20 = price_data["price"] * random.uniform(0.95, 1.05)
                rsi = random.uniform(25, 75)
                macd = random.uniform(-2, 2)
                signal = "🟢 Bullish" if macd > 0 else "🔴 Bearish"

                result += f"{symbol:<8} ${sma_20:<9.2f} {rsi:<8.1f} {macd:<10.2f} {signal:<15}\n"

        elif comparison_type == "risk":
            result += f"{'Symbol':<8} {'Beta':<8} {'Vol%':<10} {'Risk':<15} {'VaR 95%':<12}\n"
            result += "─" * 53 + "\n"

            for symbol in valid_symbols:
                stock_info = MOCK_STOCK_DATA[symbol]
                price_data = _generate_realistic_price(symbol)
                volatility = float(stock_info["volatility"]) * 100
                beta = random.uniform(0.7, 1.8)
                risk_level = "High" if volatility > 3 else "Moderate" if volatility > 2 else "Low"
                var_95 = price_data["price"] * volatility * 1.65 / 100

                result += f"{symbol:<8} {beta:<8.2f} {volatility:<10.2f}% {risk_level:<15} ${var_95:<11.2f}\n"

        result += "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        result += RISK_DISCLAIMER()

        return result

    except Exception as e:
        return f"❌ Error comparing stocks: {str(e)}"
