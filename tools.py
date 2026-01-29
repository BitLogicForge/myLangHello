"""Agent tools - all utility functions with LangChain tool decorators."""

from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import requests
import platform
import random
import ast
import operator

load_dotenv()


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression safely and return the result as a string.
    Supports basic operations: +, -, *, /, ** (power)
    Example: "2 + 2 * (3 - 1)"
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
                    raise ValueError(f"Unsupported operation: {type(node.op).__name__}")
                return allowed_ops[type(node.op)](eval_node(node.left), eval_node(node.right))
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


@tool
def weather(city: str) -> str:
    """Return a fake weather report for the given city."""
    temp_c = random.randint(-10, 35)
    possible_conditions = ["sunny", "cloudy", "rainy", "windy", "snowy"]
    condition = random.choice(possible_conditions)
    return f"The weather in {city} is {condition} and {temp_c}°C."


@tool
def read_file(path: str) -> str:
    """Read a file and return its contents or an error message."""
    try:
        if not os.path.exists(path):
            return f"Error: file not found: {path}"
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


@tool
def write_file(spec: str) -> str:
    """Write content to a file. Spec: first line is path, rest is content."""
    try:
        if "\n" not in spec:
            return "Error: write spec must include a path line, then a newline, then content"
        path, content = spec.split("\n", 1)
        dirpath = os.path.dirname(path)
        if dirpath and not os.path.exists(dirpath):
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Wrote {len(content)} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


@tool
def current_date(with_date: bool = True, with_time: bool = False) -> str:
    """Return the current date and/or time as a string.

    Args:
        with_date: Include date (default: True)
        with_time: Include time (default: False)

    Returns date only, time only, or both based on the parameters.
    """
    from datetime import datetime

    now = datetime.now()

    if with_date and with_time:
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif with_date:
        return now.strftime("%Y-%m-%d")
    elif with_time:
        return now.strftime("%H:%M:%S")
    else:
        return now.strftime("%Y-%m-%d")  # Default to date if both are False


@tool
def http_get(url: str) -> str:
    """Perform an HTTP GET and return a short summary/result."""
    try:
        resp = requests.get(url, timeout=5)
        summary = f"Status: {resp.status_code}; Length: {len(resp.content)}"
        try:
            text_preview = resp.text[:1000]
            return summary + "\n" + text_preview
        except Exception:
            return summary
    except Exception as e:
        return f"HTTP GET error: {e}"


@tool
def list_dir(path: str = ".") -> str:
    """List files in a directory and return a newline-separated listing."""
    try:
        if not os.path.exists(path):
            return f"Error: path not found: {path}"
        items = os.listdir(path)
        return "\n".join(items)
    except Exception as e:
        return f"Error listing directory: {e}"


@tool
def system_info(query: str = "") -> str:
    """Return basic system information. The query parameter is ignored."""
    return f"{platform.system()} {platform.release()} ({platform.machine()})"


@tool
def random_joke(query: str = "") -> str:
    """Return a small, harmless random joke. The query parameter is ignored."""
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
    return random.choice(jokes)


@tool
def joke_format(joke: str) -> str:
    """Format a joke with decorative borders for better presentation.

    Args:
        joke: The joke text to format

    Returns a nicely formatted joke with visual separators.Do not add any extra text.
    """
    border = "═" * 50
    return f"""
╔{border}╗
║  Best joke for you:
║
║  {joke}
╚{border}╝
"""


@tool
def loan_calculator(principal: float, annual_rate: float, years: int) -> str:
    """Calculate loan payments given principal, annual rate, and term in years."""
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


@tool
def currency_converter(amount: float, from_currency: str, to_currency: str) -> str:
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
