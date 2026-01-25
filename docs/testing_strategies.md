# Testing Strategies for AI Agents

Comprehensive guide to testing LangChain agents for reliability and quality.

---

## Table of Contents

1. [Overview](#overview)
2. [Testing Pyramid](#testing-pyramid)
3. [Unit Testing](#unit-testing)
4. [Integration Testing](#integration-testing)
5. [Evaluation & Benchmarking](#evaluation--benchmarking)
6. [Test Automation](#test-automation)
7. [Production Examples](#production-examples)

---

## Overview

### Why Test AI Agents?

**Challenges:**

- Non-deterministic outputs
- Complex tool interactions
- Multiple failure points
- Prompt sensitivity
- Context dependencies

**Testing Goals:**

- ✅ Verify tool functionality
- ✅ Ensure correct tool selection
- ✅ Validate output quality
- ✅ Catch regressions
- ✅ Monitor performance

---

## Testing Pyramid

```
        /\
       /  \        E2E Tests (Few)
      /____\       - Full agent workflows
     /      \      - Real LLM calls
    /________\     - Slow, expensive
   /          \
  / Integration \   Integration Tests (Some)
 /______________\  - Agent + tools
/                \ - May mock LLM
/   Unit Tests    \ Unit Tests (Many)
/_____(Tools)_____\- Individual tools
                   - Fast, deterministic
```

---

## Unit Testing

### Testing Individual Tools

```python
# tests/test_tools.py
import pytest
from tools.base_tools import calculator, safe_read_file

class TestCalculator:
    """Unit tests for calculator tool."""

    def test_basic_addition(self):
        """Test simple addition."""
        result = calculator.invoke({"expression": "5 + 3"})
        assert result == "8"

    def test_multiplication(self):
        """Test multiplication."""
        result = calculator.invoke({"expression": "4 * 7"})
        assert result == "28"

    def test_division(self):
        """Test division."""
        result = calculator.invoke({"expression": "10 / 2"})
        assert result == "5.0"

    def test_division_by_zero(self):
        """Test error handling for division by zero."""
        result = calculator.invoke({"expression": "10 / 0"})
        assert "Error" in result
        assert "divide by zero" in result.lower()

    def test_invalid_expression(self):
        """Test error handling for invalid input."""
        result = calculator.invoke({"expression": "5 ++ 3"})
        assert "Error" in result

    def test_unsafe_expression(self):
        """Test rejection of unsafe code."""
        result = calculator.invoke({"expression": "__import__('os')"})
        assert "Error" in result
        assert "unsafe" in result.lower()


class TestFileReader:
    """Unit tests for file reading tool."""

    @pytest.fixture
    def temp_file(self, tmp_path):
        """Create a temporary file for testing."""
        file_path = tmp_path / "test.txt"
        file_path.write_text("Hello, World!")
        return file_path

    def test_read_existing_file(self, temp_file):
        """Test reading a file that exists."""
        result = safe_read_file.invoke({"path": str(temp_file)})
        assert result == "Hello, World!"

    def test_read_nonexistent_file(self):
        """Test reading a file that doesn't exist."""
        result = safe_read_file.invoke({"path": "nonexistent.txt"})
        assert "Error" in result
        assert "not found" in result.lower()

    def test_invalid_path(self):
        """Test rejection of invalid paths."""
        result = safe_read_file.invoke({"path": "../../../etc/passwd"})
        assert "Error" in result
        assert "Invalid" in result

    def test_large_file_limit(self, tmp_path):
        """Test file size limit."""
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 200000)  # 200KB

        result = safe_read_file.invoke({"path": str(large_file)})
        assert "Error" in result
        assert "too large" in result.lower()


# Run tests
# pytest tests/test_tools.py -v
```

---

### Testing Tool Selection Logic

```python
# tests/test_tool_selection.py
import pytest
from unittest.mock import Mock, patch
from agents.base_agent import AgentFactory

class TestToolSelection:
    """Test that agent selects correct tools."""

    @pytest.fixture
    def mock_llm(self):
        """Mock LLM for predictable responses."""
        llm = Mock()
        # Configure mock to return specific tool calls
        return llm

    def test_selects_calculator_for_math(self, mock_llm):
        """Agent should use calculator for math questions."""
        factory = AgentFactory()

        # Mock the LLM to return calculator tool call
        with patch('langchain_openai.ChatOpenAI') as mock_chat:
            mock_chat.return_value = mock_llm

            agent = factory.create_base_agent()

            # Verify calculator tool is available
            tool_names = [tool.name for tool in agent.tools]
            assert 'calculator' in tool_names

    def test_selects_database_for_data_queries(self):
        """Agent should use database for data questions."""
        # Implementation depends on your agent setup
        pass


# Parametrized tests for multiple scenarios
@pytest.mark.parametrize("input_text,expected_tool", [
    ("What's 5 + 3?", "calculator"),
    ("Read file data.txt", "read_file"),
    ("Search for Python tutorials", "web_search"),
    ("How many users?", "sql_db_query"),
])
def test_tool_selection_scenarios(input_text, expected_tool):
    """Test tool selection for various inputs."""
    # This would require mocking or actual agent execution
    # Implementation depends on your testing strategy
    pass
```

---

## Integration Testing

### Testing Agent with Mock LLM

```python
# tests/test_agent_integration.py
import pytest
from langchain_core.messages import AIMessage, FunctionMessage
from langchain_core.language_models import FakeListChatModel
from agents.base_agent import AgentFactory

class TestAgentIntegration:
    """Integration tests with mock LLM."""

    @pytest.fixture
    def mock_responses(self):
        """Predefined LLM responses for testing."""
        return [
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "calculator",
                        "arguments": '{"expression": "5 + 3"}'
                    }
                }
            ),
            AIMessage(content="The answer is 8")
        ]

    def test_agent_uses_calculator(self, mock_responses):
        """Test agent workflow with calculator tool."""
        # Create mock LLM with predefined responses
        mock_llm = FakeListChatModel(responses=mock_responses)

        factory = AgentFactory()
        # Create agent with mock LLM
        agent_executor = factory.create_base_agent()

        # Replace LLM with mock
        agent_executor.agent.llm = mock_llm

        result = agent_executor.invoke({"input": "What's 5 + 3?"})

        assert "8" in result["output"]

    def test_agent_handles_tool_error(self):
        """Test agent handles tool failures gracefully."""
        mock_responses = [
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "read_file",
                        "arguments": '{"path": "nonexistent.txt"}'
                    }
                }
            ),
            AIMessage(content="I couldn't find that file.")
        ]

        mock_llm = FakeListChatModel(responses=mock_responses)

        factory = AgentFactory()
        agent_executor = factory.create_base_agent()
        agent_executor.agent.llm = mock_llm

        result = agent_executor.invoke({"input": "Read nonexistent.txt"})

        # Should handle error gracefully
        assert "Error" in result["intermediate_steps"][0][1] or \
               "couldn't find" in result["output"].lower()


class TestAgentWithRealLLM:
    """Integration tests with real LLM (slower, more expensive)."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_math_calculation(self):
        """Test real agent performs calculation."""
        factory = AgentFactory()
        agent = factory.create_base_agent()

        result = agent.invoke({"input": "Calculate 15% of 200"})

        # Check that answer is correct
        assert "30" in result["output"]

    @pytest.mark.integration
    @pytest.mark.slow
    def test_multi_step_reasoning(self):
        """Test agent can handle multi-step problems."""
        factory = AgentFactory()
        agent = factory.create_base_agent()

        result = agent.invoke({
            "input": "Calculate 10 + 5, then multiply the result by 2"
        })

        # Should use calculator twice and get 30
        assert "30" in result["output"]


# Run integration tests only when specified
# pytest tests/test_agent_integration.py -m integration
```

---

### Testing Memory and Context

```python
# tests/test_memory.py
import pytest
from langchain.memory import ConversationBufferMemory
from agents.base_agent import AgentFactory

class TestMemory:
    """Test conversation memory."""

    def test_memory_stores_messages(self):
        """Test that memory stores conversation."""
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Simulate conversation
        memory.save_context(
            {"input": "My name is John"},
            {"output": "Nice to meet you, John!"}
        )

        # Check memory
        messages = memory.load_memory_variables({})["chat_history"]
        assert len(messages) == 2
        assert "John" in messages[0].content

    def test_agent_remembers_context(self):
        """Test agent uses memory across turns."""
        factory = AgentFactory()
        agent = factory.create_base_agent()

        # First message
        result1 = agent.invoke({"input": "My name is Alice"})

        # Second message - should remember name
        result2 = agent.invoke({"input": "What's my name?"})

        assert "Alice" in result2["output"]

    def test_memory_persistence(self, tmp_path):
        """Test memory can be saved and loaded."""
        from memory.manager import FilePersistedMemory

        session_id = "test_session"
        memory_dir = str(tmp_path)

        # Create and save memory
        memory1 = FilePersistedMemory(session_id, memory_dir)
        memory1.memory.save_context(
            {"input": "Test message"},
            {"output": "Test response"}
        )
        memory1.save()

        # Load in new instance
        memory2 = FilePersistedMemory(session_id, memory_dir)
        messages = memory2.memory.load_memory_variables({})["chat_history"]

        assert len(messages) == 2
        assert "Test message" in messages[0].content
```

---

## Evaluation & Benchmarking

### Creating Test Datasets

```python
# tests/test_datasets.py

# Define test cases with expected outputs
MATH_TEST_CASES = [
    {
        "input": "What's 5 + 3?",
        "expected_answer": "8",
        "expected_tool": "calculator"
    },
    {
        "input": "Calculate 15% of 200",
        "expected_answer": "30",
        "expected_tool": "calculator"
    },
    {
        "input": "What's 100 divided by 4?",
        "expected_answer": "25",
        "expected_tool": "calculator"
    }
]

DATABASE_TEST_CASES = [
    {
        "input": "How many users are in the database?",
        "expected_tool": "sql_db_query",
        "should_contain": ["SELECT", "COUNT", "users"]
    },
    {
        "input": "Show me the top 5 products by sales",
        "expected_tool": "sql_db_query",
        "should_contain": ["SELECT", "ORDER BY", "LIMIT 5"]
    }
]

FILE_TEST_CASES = [
    {
        "input": "Read the file config.json",
        "expected_tool": "read_file",
        "expected_behavior": "attempt_read"
    }
]
```

---

### Automated Evaluation

```python
# tests/evaluator.py
from typing import List, Dict
import json
from dataclasses import dataclass
from datetime import datetime

@dataclass
class EvaluationResult:
    """Result of a single test case."""
    input: str
    expected: str
    actual: str
    passed: bool
    execution_time: float
    tokens_used: int
    error: str = None

class AgentEvaluator:
    """Automated evaluation of agent performance."""

    def __init__(self, agent_executor):
        self.agent_executor = agent_executor
        self.results: List[EvaluationResult] = []

    def evaluate_test_case(self, test_case: Dict) -> EvaluationResult:
        """Evaluate a single test case."""
        import time
        from langchain.callbacks import get_openai_callback

        start_time = time.time()

        try:
            with get_openai_callback() as cb:
                response = self.agent_executor.invoke({
                    "input": test_case["input"]
                })

                output = response.get("output", "")
                expected = test_case.get("expected_answer", "")

                # Check if expected answer is in output
                passed = expected.lower() in output.lower()

                result = EvaluationResult(
                    input=test_case["input"],
                    expected=expected,
                    actual=output,
                    passed=passed,
                    execution_time=time.time() - start_time,
                    tokens_used=cb.total_tokens
                )

        except Exception as e:
            result = EvaluationResult(
                input=test_case["input"],
                expected=test_case.get("expected_answer", ""),
                actual="",
                passed=False,
                execution_time=time.time() - start_time,
                tokens_used=0,
                error=str(e)
            )

        self.results.append(result)
        return result

    def evaluate_dataset(self, test_cases: List[Dict]) -> Dict:
        """Evaluate multiple test cases."""
        print(f"Evaluating {len(test_cases)} test cases...")

        for i, test_case in enumerate(test_cases, 1):
            print(f"Test {i}/{len(test_cases)}: {test_case['input'][:50]}...")
            self.evaluate_test_case(test_case)

        return self.generate_report()

    def generate_report(self) -> Dict:
        """Generate evaluation report."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed

        avg_time = sum(r.execution_time for r in self.results) / total if total > 0 else 0
        avg_tokens = sum(r.tokens_used for r in self.results) / total if total > 0 else 0

        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": (passed / total * 100) if total > 0 else 0,
            "avg_execution_time": avg_time,
            "avg_tokens_used": avg_tokens,
            "failed_cases": [
                {
                    "input": r.input,
                    "expected": r.expected,
                    "actual": r.actual,
                    "error": r.error
                }
                for r in self.results if not r.passed
            ]
        }

        return report

    def save_report(self, filename: str = "evaluation_report.json"):
        """Save report to file."""
        report = self.generate_report()

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"\nReport saved to {filename}")
        print(f"Pass Rate: {report['pass_rate']:.1f}%")
        print(f"Avg Time: {report['avg_execution_time']:.2f}s")
        print(f"Avg Tokens: {report['avg_tokens_used']:.0f}")

# Usage
evaluator = AgentEvaluator(agent_executor)
report = evaluator.evaluate_dataset(MATH_TEST_CASES)
evaluator.save_report("math_evaluation.json")
```

---

### Regression Testing

```python
# tests/test_regression.py
import pytest
import json
from pathlib import Path

class TestRegression:
    """Prevent performance regressions."""

    @pytest.fixture
    def baseline_metrics(self):
        """Load baseline performance metrics."""
        baseline_file = Path("tests/baseline_metrics.json")

        if baseline_file.exists():
            with open(baseline_file) as f:
                return json.load(f)

        return {
            "pass_rate": 90.0,
            "avg_tokens": 500,
            "avg_time": 3.0
        }

    def test_pass_rate_not_degraded(self, baseline_metrics):
        """Ensure pass rate hasn't degraded."""
        evaluator = AgentEvaluator(agent_executor)
        report = evaluator.evaluate_dataset(MATH_TEST_CASES)

        baseline_pass_rate = baseline_metrics["pass_rate"]
        current_pass_rate = report["pass_rate"]

        # Allow 5% degradation tolerance
        assert current_pass_rate >= baseline_pass_rate - 5, \
            f"Pass rate degraded: {current_pass_rate}% < {baseline_pass_rate}%"

    def test_token_usage_not_increased(self, baseline_metrics):
        """Ensure token usage hasn't significantly increased."""
        evaluator = AgentEvaluator(agent_executor)
        report = evaluator.evaluate_dataset(MATH_TEST_CASES)

        baseline_tokens = baseline_metrics["avg_tokens"]
        current_tokens = report["avg_tokens_used"]

        # Allow 20% increase tolerance
        assert current_tokens <= baseline_tokens * 1.2, \
            f"Token usage increased: {current_tokens} > {baseline_tokens * 1.2}"

    def save_new_baseline(self):
        """Save current metrics as new baseline."""
        evaluator = AgentEvaluator(agent_executor)
        report = evaluator.evaluate_dataset(MATH_TEST_CASES)

        baseline = {
            "pass_rate": report["pass_rate"],
            "avg_tokens": report["avg_tokens_used"],
            "avg_time": report["avg_execution_time"]
        }

        with open("tests/baseline_metrics.json", 'w') as f:
            json.dump(baseline, f, indent=2)
```

---

## Test Automation

### GitHub Actions CI/CD

```yaml
# .github/workflows/test.yml
name: Test Agent

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov

      - name: Run unit tests
        run: |
          pytest tests/test_tools.py -v

      - name: Run integration tests
        run: |
          pytest tests/test_agent_integration.py -m "not slow" -v
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}

      - name: Check test coverage
        run: |
          pytest --cov=tools --cov=agents --cov-report=html

      - name: Upload coverage report
        uses: codecov/codecov-action@v2
```

---

### Pre-commit Hooks

```python
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/test_tools.py -v
        language: system
        pass_filenames: false
        always_run: true
```

---

## Production Examples

### Complete Test Suite

```python
# tests/conftest.py
import pytest
from agents.base_agent import AgentFactory
from config.settings import Settings

@pytest.fixture(scope="session")
def test_settings():
    """Test-specific settings."""
    return Settings(
        openai_api_key="test-key",
        model_name="gpt-3.5-turbo",
        verbose=False,
        max_iterations=5
    )

@pytest.fixture
def agent_factory(test_settings):
    """Agent factory for tests."""
    return AgentFactory()

@pytest.fixture
def mock_agent_executor(agent_factory):
    """Mock agent executor."""
    return agent_factory.create_base_agent()


# tests/test_complete_workflow.py
import pytest

class TestCompleteWorkflow:
    """End-to-end workflow tests."""

    @pytest.mark.e2e
    def test_customer_support_workflow(self, agent_executor):
        """Test complete customer support interaction."""
        # Step 1: Greet customer
        response1 = agent_executor.invoke({
            "input": "Hello, I need help with my order"
        })
        assert "help" in response1["output"].lower()

        # Step 2: Provide order number
        response2 = agent_executor.invoke({
            "input": "My order number is 12345"
        })
        # Should query database or acknowledge

        # Step 3: Check status
        response3 = agent_executor.invoke({
            "input": "What's the status?"
        })
        # Should provide status

        # Verify conversation flow maintained context
        assert "12345" in str(response3) or "order" in response3["output"].lower()
```

---

## Best Practices

### 1. Test Pyramid Distribution

```
- 70% Unit tests (tools)
- 20% Integration tests (agent + tools)
- 10% E2E tests (full workflows)
```

### 2. Use Fixtures for Reusability

```python
@pytest.fixture
def sample_agent():
    """Reusable agent fixture."""
    return create_test_agent()
```

### 3. Mock External Dependencies

```python
@patch('requests.get')
def test_api_tool(mock_get):
    mock_get.return_value.json.return_value = {"data": "test"}
    # Test tool
```

### 4. Separate Fast/Slow Tests

```python
@pytest.mark.slow
def test_with_real_llm():
    pass

# Run only fast tests
# pytest -m "not slow"
```

### 5. Track Metrics Over Time

```python
# Save evaluation results
evaluator.save_report(f"reports/{datetime.now()}.json")
```

---

_Last Updated: January 25, 2026_
