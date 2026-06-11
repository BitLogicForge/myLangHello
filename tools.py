"""Agent tools - all utility functions with LangChain tool decorators."""

import ast
import asyncio
import operator
import os
import random
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from pydantic import BaseModel, Field

load_dotenv()

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
# MARK: calculator tool
# ==========================================

class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="The mathematical expression to evaluate safely (e.g. '2 + 2 * (3 - 1)'). Supports basic operations: +, -, *, /, **"
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
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.USub: operator.neg,
        }

        def eval_node(node):
            if isinstance(node, ast.Num):  # Number
                return node.n
            elif isinstance(node, ast.Constant):  # Python 3.8+ uses Constant
                return node.value
            elif isinstance(node, ast.BinOp):  # Binary operation
                if type(node.op) not in allowed_ops:
                    raise ValueError(
                        f"Unsupported operation: {type(node.op).__name__}")
                return allowed_ops[type(node.op)](eval_node(node.left), eval_node(node.right))
            elif isinstance(node, ast.UnaryOp):  # Unary operation (e.g., -5)
                if type(node.op) not in allowed_ops:
                    raise ValueError(
                        f"Unsupported operation: {type(node.op).__name__}")
                return allowed_ops[type(node.op)](eval_node(node.operand))
            else:
                raise ValueError(
                    f"Unsupported expression type: {type(node).__name__}")

        tree = ast.parse(expression, mode="eval")
        result = eval_node(tree.body)
        return f"Result: {result}"
    except (SyntaxError, ValueError) as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Error: {e}"


# ==========================================
# MARK: weather tool
# ==========================================

class WeatherInput(BaseModel):
    city: str = Field(
        ...,
        description="The city name to get weather for (e.g. 'Poznan', 'London')."
    )


@tool(args_schema=WeatherInput)
async def weather(city: str) -> str:
    """Return a fake weather report for the given city."""
    temp_c = random.randint(-10, 35)
    possible_conditions = ["sunny", "cloudy", "rainy", "windy", "snowy"]
    condition = random.choice(possible_conditions)
    return f"The weather in {city} is {condition} and {temp_c}°C."


# ==========================================
# MARK: read file tools
# ==========================================

class ReadFileInput(BaseModel):
    path: str = Field(
        ...,
        description="The path of the file to read (relative to the safe sandbox folder)."
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
# MARK: write file tool
# ==========================================

class WriteFileInput(BaseModel):
    path: str = Field(
        ...,
        description="The path of the file to write to (relative to the safe sandbox folder)."
    )
    content: str = Field(
        ...,
        description="The text content to write into the file."
    )


@tool(args_schema=WriteFileInput)
async def write_file(path: str, content: str) -> str:
    """Write content to a file in the sandbox directory. Return success or error message."""
    try:
        safe_p = _safe_path(path)
        await asyncio.to_thread(safe_p.parent.mkdir, parents=True, exist_ok=True)
        await asyncio.to_thread(safe_p.write_text, content, encoding="utf-8")
        return f"Wrote {len(content)} bytes to sandbox file: {path}"
    except PermissionError as pe:
        return str(pe)
    except Exception as e:
        return f"Error writing file: {e}"


# ==========================================
# MARK: current date/time tool
# ==========================================

class CurrentDateInput(BaseModel):
    with_date: bool = Field(
        default=True,
        description="Include date (defaults to True)."
    )
    with_time: bool = Field(
        default=False,
        description="Include time (defaults to False)."
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
# MARK: http get tool
# ==========================================

class HttpGetInput(BaseModel):
    url: str = Field(
        ...,
        description="The HTTP/HTTPS URL to perform a GET request on."
    )


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
# MARK: random joke tool
# ==========================================

class RandomJokeInput(BaseModel):
    query: str = Field(
        default="",
        description="Optional keyword search query to filter jokes by topic (e.g. 'bug', 'Java')."
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

    selected_joke = random.choice(
        filtered_jokes) if filtered_jokes else random.choice(jokes)

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
# MARK: joke formatting tool
# ==========================================

class JokeFormatInput(BaseModel):
    joke: str = Field(
        ...,
        description="The raw joke text to format with decorative borders."
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
# MARK: loan calculator tool
# ==========================================

class LoanCalculatorInput(BaseModel):
    principal: float = Field(
        ...,
        gt=0,
        description="The principal loan amount in USD (must be positive)."
    )
    annual_rate: float = Field(
        ...,
        ge=0,
        description="The annual interest rate as a percentage (e.g., 5.5 for 5.5%)."
    )
    years: int = Field(
        ...,
        gt=0,
        description="The term of the loan in years (must be positive)."
    )


@tool(args_schema=LoanCalculatorInput)
async def loan_calculator(principal: float, annual_rate: float, years: int) -> str:
    """Calculate loan payments given principal in USD, annual rate, and term in years."""
    try:
        if principal <= 0 or annual_rate < 0 or years <= 0:
            return "Error: Principal and years must be positive, rate must be non-negative"

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
Interest Percentage: {(total_interest/principal)*100:.2f}% of principal
"""
        return result
    except ValueError:
        return "Error: Invalid number format. Use numbers only (e.g., '200000,5.5,30')"
    except Exception as e:
        return f"Error calculating loan: {e}"


# ==========================================
# MARK: currency converter tool
# ==========================================

class CurrencyConverterInput(BaseModel):
    amount: float = Field(
        ...,
        gt=0,
        description="The currency amount to convert (must be positive)."
    )
    from_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="The 3-letter currency code to convert from (e.g., 'USD')."
    )
    to_currency: str = Field(
        ...,
        min_length=3,
        max_length=3,
        description="The 3-letter currency code to convert to (e.g., 'EUR')."
    )


@tool(args_schema=CurrencyConverterInput)
async def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
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
            return f"Error: '{from_curr}' not supported. Available currencies: {available}"

        if to_curr not in exchange_rates:
            available = ", ".join(sorted(exchange_rates.keys()))
            return f"Error: '{to_curr}' not supported. Available currencies: {available}"

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
# MARK: city to coordinates tool
# ==========================================

class CityToCoordinatesInput(BaseModel):
    city: str = Field(
        ...,
        description="The city name to find coordinates for (e.g. 'Paris', 'New York')."
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

        data = resp.json()
        results = data.get("results")
        if not results:
            return f"Error: City '{city}' not found."

        loc = results[0]
        name = loc.get("name")
        country = loc.get("country", "Unknown")
        lat = loc.get("latitude")
        lon = loc.get("longitude")
        timezone = loc.get("timezone", "Unknown")

        return f"City: {name}, Country: {country}, Latitude: {lat}, Longitude: {lon}, Timezone: {timezone}"
    except Exception as e:
        return f"Error finding coordinates: {e}"
