# Error Handling & Retry Logic Guide

Comprehensive guide to handling failures gracefully in LangChain agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Common Error Types](#common-error-types)
3. [Error Handling Strategies](#error-handling-strategies)
4. [Retry Mechanisms](#retry-mechanisms)
5. [Fallback Patterns](#fallback-patterns)
6. [Circuit Breakers](#circuit-breakers)
7. [Production Examples](#production-examples)

---

## Overview

### Why Error Handling Matters

**Problems in Production:**

- LLM API timeouts or rate limits
- Tool execution failures
- Network issues
- Invalid tool outputs
- Parsing errors
- Context window exceeded

**Solution:** Implement robust error handling with:

- Graceful degradation
- Automatic retries
- Fallback mechanisms
- User-friendly error messages
- Logging and monitoring

---

## Common Error Types

### 1. LLM API Errors

```python
# Rate limit exceeded
openai.error.RateLimitError: Rate limit reached for gpt-4

# Timeout
openai.error.Timeout: Request timed out

# Authentication
openai.error.AuthenticationError: Invalid API key

# Context length exceeded
openai.error.InvalidRequestError: This model's maximum context length is 8192 tokens
```

### 2. Tool Execution Errors

```python
# File not found
FileNotFoundError: [Errno 2] No such file or directory: 'data.csv'

# Database connection
sqlalchemy.exc.OperationalError: could not connect to server

# API call failure
requests.exceptions.ConnectionError: Connection refused

# Permission denied
PermissionError: [Errno 13] Permission denied
```

### 3. Agent Parsing Errors

```python
# Invalid JSON from LLM
json.JSONDecodeError: Expecting value: line 1 column 1 (char 0)

# Malformed tool call
OutputParserException: Could not parse LLM output
```

---

## Error Handling Strategies

### Strategy 1: Try-Except in Tools

**Wrap each tool with error handling**

```python
from langchain_core.tools import tool
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@tool
def safe_calculator(expression: str) -> str:
    """Safely evaluate a math expression."""
    try:
        # Validate input first
        if any(char in expression for char in ['__', 'import', 'eval', 'exec']):
            return "Error: Invalid expression - unsafe characters detected"

        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except ZeroDivisionError:
        logger.error(f"Division by zero: {expression}")
        return "Error: Cannot divide by zero"

    except SyntaxError as e:
        logger.error(f"Syntax error in expression '{expression}': {e}")
        return f"Error: Invalid mathematical expression. {str(e)}"

    except Exception as e:
        logger.error(f"Unexpected error in calculator: {e}", exc_info=True)
        return f"Error: Could not evaluate expression. {str(e)}"


@tool
def safe_read_file(path: str) -> str:
    """Safely read a file with error handling."""
    try:
        # Validate path
        if '..' in path or path.startswith('/'):
            return "Error: Invalid file path"

        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Limit file size
        if len(content) > 100000:
            return "Error: File too large (max 100KB)"

        return content

    except FileNotFoundError:
        logger.warning(f"File not found: {path}")
        return f"Error: File '{path}' not found"

    except PermissionError:
        logger.error(f"Permission denied: {path}")
        return f"Error: No permission to read '{path}'"

    except UnicodeDecodeError:
        logger.error(f"Cannot decode file: {path}")
        return "Error: File is not a text file or has invalid encoding"

    except Exception as e:
        logger.error(f"Error reading file '{path}': {e}", exc_info=True)
        return f"Error: Could not read file. {str(e)}"
```

---

### Strategy 2: AgentExecutor Error Handling

**Configure built-in error handling**

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,

    # Error handling configuration
    handle_parsing_errors=True,  # Handle LLM parsing errors
    max_iterations=10,            # Prevent infinite loops
    max_execution_time=120,       # Timeout after 2 minutes

    # Custom error message
    handle_parsing_errors="Check your output and try again. Make sure to use the correct format.",
)
```

**Custom parsing error handler:**

```python
def custom_parsing_error_handler(error: Exception) -> str:
    """Custom handler for parsing errors."""
    logger.error(f"Parsing error: {error}")

    return (
        "I encountered an error processing that request. "
        "Let me try a different approach. "
        "Please use the tools exactly as described."
    )

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=custom_parsing_error_handler
)
```

---

### Strategy 3: Wrapper with Comprehensive Error Handling

**Production-ready agent wrapper**

```python
from typing import Dict, Any, Optional
from datetime import datetime
import traceback

class RobustAgentExecutor:
    """Agent executor with comprehensive error handling."""

    def __init__(self, agent_executor: AgentExecutor):
        self.agent_executor = agent_executor
        self.error_count = 0
        self.max_errors = 3

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with error handling."""

        try:
            # Reset error count on new conversation
            self.error_count = 0

            # Execute agent
            response = self.agent_executor.invoke(inputs)

            return {
                "success": True,
                "output": response.get("output", ""),
                "timestamp": datetime.utcnow().isoformat()
            }

        except openai.error.RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e}")
            return {
                "success": False,
                "error": "rate_limit",
                "message": "Service is experiencing high demand. Please try again in a moment.",
                "retry_after": 60
            }

        except openai.error.Timeout as e:
            logger.error(f"Request timeout: {e}")
            return {
                "success": False,
                "error": "timeout",
                "message": "Request took too long. Please try a simpler query."
            }

        except openai.error.InvalidRequestError as e:
            logger.error(f"Invalid request: {e}")

            if "maximum context length" in str(e):
                return {
                    "success": False,
                    "error": "context_length",
                    "message": "Conversation is too long. Please start a new conversation."
                }

            return {
                "success": False,
                "error": "invalid_request",
                "message": "Invalid request. Please try rephrasing your question."
            }

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)

            self.error_count += 1

            if self.error_count >= self.max_errors:
                return {
                    "success": False,
                    "error": "max_errors",
                    "message": "Multiple errors occurred. Please start a new conversation."
                }

            return {
                "success": False,
                "error": "unknown",
                "message": "An unexpected error occurred. Please try again.",
                "details": str(e) if self.agent_executor.verbose else None
            }

# Usage
robust_executor = RobustAgentExecutor(agent_executor)
result = robust_executor.invoke({"input": "What's the weather?"})

if result["success"]:
    print(result["output"])
else:
    print(f"Error: {result['message']}")
```

---

## Retry Mechanisms

### Pattern 1: Simple Retry Decorator

```python
import time
from functools import wraps
from typing import Callable, Any

def retry(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """Retry decorator with exponential backoff."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            current_delay = delay

            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)

                except (openai.error.RateLimitError, openai.error.Timeout) as e:
                    if attempt == max_attempts - 1:
                        raise

                    logger.warning(
                        f"Attempt {attempt + 1}/{max_attempts} failed: {e}. "
                        f"Retrying in {current_delay}s..."
                    )

                    time.sleep(current_delay)
                    current_delay *= backoff

                except Exception as e:
                    # Don't retry on other errors
                    raise

            return None

        return wrapper
    return decorator

# Usage
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def call_agent(agent_executor, message: str):
    """Call agent with automatic retry."""
    return agent_executor.invoke({"input": message})
```

---

### Pattern 2: Tenacity Library (Recommended)

**More sophisticated retry logic**

```python
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import logging

logger = logging.getLogger(__name__)

@retry(
    # Stop after 3 attempts
    stop=stop_after_attempt(3),

    # Exponential backoff: wait 1s, 2s, 4s
    wait=wait_exponential(multiplier=1, min=1, max=10),

    # Only retry on specific errors
    retry=retry_if_exception_type((
        openai.error.RateLimitError,
        openai.error.Timeout,
        openai.error.APIError,
    )),

    # Log before each retry
    before_sleep=before_sleep_log(logger, logging.WARNING)
)
def call_llm_with_retry(llm, messages):
    """Call LLM with automatic retry."""
    return llm(messages)


# For agent execution
@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=2, max=5),
    retry=retry_if_exception_type((
        openai.error.RateLimitError,
        openai.error.Timeout,
    ))
)
def execute_agent_with_retry(agent_executor, inputs: dict):
    """Execute agent with retry on transient errors."""
    return agent_executor.invoke(inputs)

# Usage
try:
    result = execute_agent_with_retry(agent_executor, {"input": "Hello"})
    print(result["output"])
except Exception as e:
    print(f"Failed after retries: {e}")
```

---

### Pattern 3: Per-Tool Retry Configuration

```python
from langchain_core.tools import tool

class RetryableTool:
    """Base class for tools with retry logic."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5),
        retry=retry_if_exception_type((
            requests.exceptions.Timeout,
            requests.exceptions.ConnectionError,
        ))
    )
    def _execute_with_retry(self, *args, **kwargs):
        """Override this method in subclasses."""
        raise NotImplementedError

@tool
def http_get_with_retry(url: str) -> str:
    """HTTP GET with automatic retry."""

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=5)
    )
    def _fetch():
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text[:1000]  # Limit response size

    try:
        return _fetch()
    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed after retries: {e}")
        return f"Error: Could not fetch URL. {str(e)}"
```

---

## Fallback Patterns

### Pattern 1: Model Fallback

**Try GPT-4, fallback to GPT-3.5**

```python
from langchain_openai import ChatOpenAI

class ModelFallbackExecutor:
    """Agent executor with model fallback."""

    def __init__(self, tools, prompt):
        self.tools = tools
        self.prompt = prompt

        # Primary model
        self.primary_llm = ChatOpenAI(model="gpt-4", temperature=0)

        # Fallback model
        self.fallback_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

    def invoke(self, inputs: dict) -> dict:
        """Try primary model, fallback to cheaper model on error."""

        try:
            # Try GPT-4 first
            agent = create_openai_functions_agent(
                self.primary_llm, self.tools, self.prompt
            )
            executor = AgentExecutor(agent=agent, tools=self.tools)
            return executor.invoke(inputs)

        except openai.error.RateLimitError:
            logger.warning("GPT-4 rate limited, falling back to GPT-3.5")

            # Fallback to GPT-3.5
            agent = create_openai_functions_agent(
                self.fallback_llm, self.tools, self.prompt
            )
            executor = AgentExecutor(agent=agent, tools=self.tools)
            return executor.invoke(inputs)

        except Exception as e:
            logger.error(f"Both models failed: {e}")
            raise

# Usage
fallback_executor = ModelFallbackExecutor(tools, prompt)
result = fallback_executor.invoke({"input": "Hello"})
```

---

### Pattern 2: Tool Fallback

**Try primary tool, fallback to alternative**

```python
@tool
def search_with_fallback(query: str) -> str:
    """Search with multiple fallback options."""

    # Try primary search API
    try:
        result = primary_search_api(query)
        if result:
            return result
    except Exception as e:
        logger.warning(f"Primary search failed: {e}")

    # Fallback to secondary API
    try:
        result = secondary_search_api(query)
        if result:
            return f"(Using fallback) {result}"
    except Exception as e:
        logger.warning(f"Secondary search failed: {e}")

    # Final fallback: cached/default response
    return "Unable to perform search at this time. Please try again later."


def primary_search_api(query: str) -> str:
    """Primary search implementation."""
    response = requests.get(f"https://api.search.com?q={query}", timeout=5)
    response.raise_for_status()
    return response.json()["result"]


def secondary_search_api(query: str) -> str:
    """Fallback search implementation."""
    response = requests.get(f"https://backup-api.com?q={query}", timeout=10)
    response.raise_for_status()
    return response.json()["answer"]
```

---

### Pattern 3: Graceful Degradation

**Reduce functionality instead of complete failure**

```python
class GracefulAgent:
    """Agent that degrades gracefully."""

    def __init__(self, llm, all_tools, essential_tools):
        self.llm = llm
        self.all_tools = all_tools
        self.essential_tools = essential_tools
        self.degraded_mode = False

    def invoke(self, inputs: dict) -> dict:
        """Execute with graceful degradation."""

        # Try with all tools
        if not self.degraded_mode:
            try:
                agent = create_openai_functions_agent(
                    self.llm, self.all_tools, prompt
                )
                executor = AgentExecutor(agent=agent, tools=self.all_tools)
                return executor.invoke(inputs)

            except Exception as e:
                logger.warning(f"Full agent failed, entering degraded mode: {e}")
                self.degraded_mode = True

        # Degraded mode: use only essential tools
        try:
            agent = create_openai_functions_agent(
                self.llm, self.essential_tools, prompt
            )
            executor = AgentExecutor(agent=agent, tools=self.essential_tools)

            result = executor.invoke(inputs)
            result["warning"] = "Running in degraded mode with limited functionality"
            return result

        except Exception as e:
            logger.error(f"Degraded mode also failed: {e}")
            return {
                "output": "Service is temporarily unavailable. Please try again later.",
                "error": True
            }

# Usage
essential_tools = [calculator, basic_search]  # Critical tools only
all_tools = [calculator, basic_search, advanced_api, database_tool]

agent = GracefulAgent(llm, all_tools, essential_tools)
result = agent.invoke({"input": "What's 5 + 3?"})
```

---

## Circuit Breakers

### Pattern: Prevent Cascading Failures

```python
from datetime import datetime, timedelta
from enum import Enum

class CircuitState(Enum):
    CLOSED = "closed"    # Normal operation
    OPEN = "open"        # Blocking requests
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for external services."""

    def __init__(
        self,
        failure_threshold: int = 5,
        timeout: int = 60,
        recovery_timeout: int = 30
    ):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = None
        self.last_state_change = datetime.now()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""

        # Check if circuit should transition
        self._check_state()

        if self.state == CircuitState.OPEN:
            raise Exception("Circuit breaker is OPEN - service unavailable")

        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result

        except Exception as e:
            self._on_failure()
            raise

    def _check_state(self):
        """Check and update circuit state."""

        if self.state == CircuitState.OPEN:
            # Check if timeout has passed
            if datetime.now() - self.last_state_change > timedelta(seconds=self.timeout):
                logger.info("Circuit breaker entering HALF_OPEN state")
                self.state = CircuitState.HALF_OPEN
                self.last_state_change = datetime.now()

    def _on_success(self):
        """Handle successful call."""

        if self.state == CircuitState.HALF_OPEN:
            logger.info("Circuit breaker closing - service recovered")
            self.state = CircuitState.CLOSED
            self.failure_count = 0

        self.failure_count = max(0, self.failure_count - 1)

    def _on_failure(self):
        """Handle failed call."""

        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.failure_count >= self.failure_threshold:
            if self.state != CircuitState.OPEN:
                logger.error(
                    f"Circuit breaker opening - {self.failure_count} failures"
                )
                self.state = CircuitState.OPEN
                self.last_state_change = datetime.now()

# Usage
database_circuit = CircuitBreaker(failure_threshold=3, timeout=60)

@tool
def query_database_with_circuit_breaker(query: str) -> str:
    """Database query with circuit breaker."""

    try:
        result = database_circuit.call(execute_db_query, query)
        return str(result)

    except Exception as e:
        if "Circuit breaker is OPEN" in str(e):
            return "Database is temporarily unavailable. Using cached data."

        logger.error(f"Database query failed: {e}")
        return "Error querying database"


def execute_db_query(query: str):
    """Actual database query logic."""
    # Database query implementation
    pass
```

---

## Production Example

**Complete error handling system**

```python
# utils/error_handling.py
import logging
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
from enum import Enum

logger = logging.getLogger(__name__)

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ProductionAgentExecutor:
    """Production-ready agent executor with comprehensive error handling."""

    def __init__(
        self,
        agent_executor,
        circuit_breaker: Optional[CircuitBreaker] = None,
        error_callback: Optional[Callable] = None
    ):
        self.agent_executor = agent_executor
        self.circuit_breaker = circuit_breaker or CircuitBreaker()
        self.error_callback = error_callback
        self.error_history = []

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((
            openai.error.RateLimitError,
            openai.error.Timeout,
        ))
    )
    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with full error handling."""

        try:
            # Use circuit breaker
            result = self.circuit_breaker.call(
                self._execute_agent,
                inputs
            )

            return {
                "success": True,
                "output": result.get("output", ""),
                "metadata": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "errors": []
                }
            }

        except openai.error.RateLimitError as e:
            return self._handle_error(
                error_type="rate_limit",
                severity=ErrorSeverity.MEDIUM,
                message="Rate limit exceeded. Please try again in a moment.",
                original_error=e,
                retry_after=60
            )

        except openai.error.Timeout as e:
            return self._handle_error(
                error_type="timeout",
                severity=ErrorSeverity.MEDIUM,
                message="Request timed out. Please try a simpler query.",
                original_error=e
            )

        except openai.error.InvalidRequestError as e:
            if "maximum context length" in str(e):
                return self._handle_error(
                    error_type="context_length",
                    severity=ErrorSeverity.HIGH,
                    message="Conversation too long. Please start a new conversation.",
                    original_error=e
                )

            return self._handle_error(
                error_type="invalid_request",
                severity=ErrorSeverity.MEDIUM,
                message="Invalid request. Please rephrase your question.",
                original_error=e
            )

        except Exception as e:
            if "Circuit breaker is OPEN" in str(e):
                return self._handle_error(
                    error_type="service_unavailable",
                    severity=ErrorSeverity.HIGH,
                    message="Service temporarily unavailable. Please try again later.",
                    original_error=e
                )

            return self._handle_error(
                error_type="unknown",
                severity=ErrorSeverity.CRITICAL,
                message="An unexpected error occurred.",
                original_error=e
            )

    def _execute_agent(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the agent."""
        return self.agent_executor.invoke(inputs)

    def _handle_error(
        self,
        error_type: str,
        severity: ErrorSeverity,
        message: str,
        original_error: Exception,
        retry_after: Optional[int] = None
    ) -> Dict[str, Any]:
        """Centralized error handling."""

        error_info = {
            "type": error_type,
            "severity": severity.value,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(),
            "details": str(original_error) if self.agent_executor.verbose else None
        }

        # Log error
        if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            logger.error(f"Error: {error_info}", exc_info=original_error)
        else:
            logger.warning(f"Error: {error_info}")

        # Store in history
        self.error_history.append(error_info)

        # Call callback if provided
        if self.error_callback:
            try:
                self.error_callback(error_info)
            except Exception as e:
                logger.error(f"Error callback failed: {e}")

        # Return error response
        response = {
            "success": False,
            "error": error_type,
            "message": message,
            "severity": severity.value,
            "timestamp": error_info["timestamp"]
        }

        if retry_after:
            response["retry_after"] = retry_after

        return response

# Usage
def error_notification_callback(error_info: dict):
    """Send notification on critical errors."""
    if error_info["severity"] == "critical":
        # Send to monitoring system
        send_alert(error_info)

circuit_breaker = CircuitBreaker(failure_threshold=5, timeout=60)
production_executor = ProductionAgentExecutor(
    agent_executor=agent_executor,
    circuit_breaker=circuit_breaker,
    error_callback=error_notification_callback
)

# Execute
result = production_executor.invoke({"input": "Hello"})

if result["success"]:
    print(result["output"])
else:
    print(f"Error ({result['severity']}): {result['message']}")
```

---

## Best Practices Summary

### 1. Always Handle Errors in Tools

```python
@tool
def my_tool(param: str) -> str:
    try:
        # Tool logic
        pass
    except SpecificError as e:
        return f"Error: {helpful_message}"
```

### 2. Use Retries for Transient Errors

```python
@retry(stop=stop_after_attempt(3))
def call_external_api():
    pass
```

### 3. Implement Circuit Breakers

```python
circuit = CircuitBreaker(failure_threshold=5)
result = circuit.call(risky_operation)
```

### 4. Provide User-Friendly Messages

```python
# DON'T
return str(e)

# DO
return "Service temporarily unavailable. Please try again in a moment."
```

### 5. Log Everything

```python
logger.error(f"Error details", exc_info=True)
```

---

_Last Updated: January 25, 2026_
