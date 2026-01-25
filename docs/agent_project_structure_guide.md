# Agent Project Structure Guide

Best practices for organizing complex LangChain agents across multiple files.

---

## Table of Contents

1. [Overview](#overview)
2. [Recommended File Structure](#recommended-file-structure)
3. [Key Components](#key-components)
4. [Code Organization Patterns](#code-organization-patterns)
5. [Implementation Examples](#implementation-examples)
6. [Best Practices](#best-practices)

---

## Overview

### Why Separate Agent Code?

**Benefits:**

- ✅ Better maintainability and testability
- ✅ Easier to debug and extend
- ✅ Reusable components across projects
- ✅ Clear separation of concerns
- ✅ Simpler team collaboration
- ✅ Easier dependency management

**When to Separate:**

- Multiple tools (5+ tools)
- Multiple agent types
- Complex prompts or formatting
- Database integrations
- External API calls
- Production applications

---

## Recommended File Structure

```
project/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Environment variables, API keys
│   ├── prompts.py            # All prompt templates
│   └── db_config.py          # Database configurations
│
├── tools/
│   ├── __init__.py           # Tool registry and factory
│   ├── base_tools.py         # Simple utility tools (calculator, etc.)
│   ├── sql_tools.py          # Database-specific tools
│   ├── file_tools.py         # File operations (read, write, list)
│   ├── api_tools.py          # External API calls
│   └── formatting_tools.py   # Output formatting tools
│
├── agents/
│   ├── __init__.py
│   ├── base_agent.py         # Base agent factory
│   ├── sql_agent.py          # Specialized SQL agent
│   └── multi_agent.py        # Multi-agent orchestration
│
├── chains/
│   ├── __init__.py
│   ├── formatting_chain.py   # Post-processing chains
│   └── validation_chain.py   # Output validation
│
├── models/
│   ├── __init__.py
│   ├── schemas.py            # Pydantic models
│   └── output_formats.py     # Output format definitions
│
├── utils/
│   ├── __init__.py
│   ├── database.py           # DB connection helpers
│   └── logger.py             # Logging configuration
│
├── tests/
│   ├── test_tools.py
│   ├── test_agents.py
│   └── test_chains.py
│
├── main.py                    # Entry point
├── requirements.txt
└── .env
```

---

## Key Components

### 1. Configuration Layer (`config/`)

**Purpose:** Centralize all configuration, settings, and prompts

**settings.py:**

```python
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4"
    temperature: float = 0.0
    max_iterations: int = 10
    verbose: bool = True

    db_path: str = "sqlite:///example.db"
    db_tables: list[str] = ["users", "orders", "products"]

    class Config:
        env_file = ".env"

@lru_cache()
def get_settings() -> Settings:
    return Settings()
```

**prompts.py:**

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

def get_base_agent_prompt(schema_context: str = "") -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", f"You are a helpful assistant.\n\n{schema_context}"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

def get_formatted_agent_prompt(format_category: str) -> ChatPromptTemplate:
    format_instructions = {
        "business": "Present as executive summary with ROI focus",
        "technical": "Include code snippets and technical details",
    }
    instruction = format_instructions.get(format_category, "")

    return ChatPromptTemplate.from_messages([
        ("system", f"Format: {instruction}\n\n{{schema_context}}"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
```

---

### 2. Tools Layer (`tools/`)

**Purpose:** Organize tools by functionality

**tools/**init**.py (Tool Registry):**

```python
from .base_tools import calculator, weather, system_info
from .file_tools import read_file, write_file, list_dir
from .api_tools import http_get
from .sql_tools import get_sql_tools
from .formatting_tools import get_output_format

def get_all_tools(llm=None, db=None):
    """Factory function to get all available tools."""
    tools = [
        calculator, weather, read_file, write_file,
        list_dir, http_get, system_info, get_output_format,
    ]

    if llm and db:
        tools.extend(get_sql_tools(llm, db))

    return tools
```

**tools/base_tools.py:**

```python
from langchain_core.tools import tool

@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def weather(city: str) -> str:
    """Get weather for a city."""
    return f"Weather in {city}: Sunny, 22°C"
```

**tools/sql_tools.py:**

```python
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

def get_sql_tools(llm, db_instance: SQLDatabase):
    """Create and return SQL toolkit tools."""
    toolkit = SQLDatabaseToolkit(db=db_instance, llm=llm)
    return toolkit.get_tools()
```

---

### 3. Agents Layer (`agents/`)

**Purpose:** Agent creation and configuration

**agents/base_agent.py (Factory Pattern):**

```python
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from config.settings import get_settings
from config.prompts import get_base_agent_prompt
from tools import get_all_tools
from utils.database import get_database, load_schema_context

class AgentFactory:
    """Factory for creating configured agents."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = ChatOpenAI(
            temperature=self.settings.temperature,
            model=self.settings.model_name
        )

    def create_base_agent(self, tools=None) -> AgentExecutor:
        """Create a basic agent with standard tools."""
        if tools is None:
            db = get_database(self.settings.db_path)
            tools = get_all_tools(llm=self.llm, db=db)

        schema_context = load_schema_context()
        prompt = get_base_agent_prompt(schema_context)
        agent = create_openai_functions_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.settings.verbose,
            max_iterations=self.settings.max_iterations,
        )

    def create_formatted_agent(
        self,
        format_category: str = "business"
    ) -> AgentExecutor:
        """Create an agent with specific output formatting."""
        db = get_database(self.settings.db_path)
        tools = get_all_tools(llm=self.llm, db=db)

        from config.prompts import get_formatted_agent_prompt
        prompt = get_formatted_agent_prompt(format_category)
        agent = create_openai_functions_agent(self.llm, tools, prompt)

        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=self.settings.verbose,
            max_iterations=self.settings.max_iterations,
        )
```

**agents/sql_agent.py (Specialized Agent):**

```python
from .base_agent import AgentFactory
from tools.sql_tools import get_sql_tools
from utils.database import get_database

class SQLAgent(AgentFactory):
    """Specialized agent for SQL operations only."""

    def create(self, db_uri: str = None):
        db_uri = db_uri or self.settings.db_path
        db = get_database(db_uri)
        sql_tools = get_sql_tools(self.llm, db_instance=db)

        return self.create_base_agent(tools=sql_tools)
```

---

### 4. Utils Layer (`utils/`)

**Purpose:** Shared utilities and helpers

**utils/database.py:**

```python
import json
from functools import lru_cache
from langchain_community.utilities import SQLDatabase

@lru_cache()
def get_database(db_uri: str) -> SQLDatabase:
    """Get cached database connection."""
    return SQLDatabase.from_uri(
        db_uri,
        include_tables=["users", "orders", "products"],
        sample_rows_in_table_info=2,
    )

def load_schema_context(config_file: str = "db_schema_config.json") -> str:
    """Load and format database schema context."""
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            custom_table_info = json.load(f)
        return "\n".join([
            f"- {table}: {desc}"
            for table, desc in custom_table_info.items()
        ])
    except FileNotFoundError:
        return ""
```

---

### 5. Entry Point (`main.py`)

**Purpose:** Application entry and usage examples

```python
from agents.base_agent import AgentFactory
from agents.sql_agent import SQLAgent

def main():
    """Basic agent usage."""
    factory = AgentFactory()
    agent = factory.create_base_agent()

    question = "How many users are registered?"
    response = agent.invoke({"input": question})
    print(response["output"])

def main_formatted():
    """Formatted output agent."""
    factory = AgentFactory()
    agent = factory.create_formatted_agent(format_category="business")

    question = "Analyze the database and provide a business report."
    response = agent.invoke({"input": question})
    print(response["output"])

def main_sql_specialized():
    """SQL-only agent."""
    sql_agent_factory = SQLAgent()
    agent = sql_agent_factory.create()

    question = "What are the top 5 products by sales?"
    response = agent.invoke({"input": question})
    print(response["output"])

if __name__ == "__main__":
    main()
```

---

## Code Organization Patterns

### 1. Factory Pattern

**Why:** Create agents with different configurations without duplication

```python
# Instead of duplicating agent creation code:
factory = AgentFactory()
agent1 = factory.create_base_agent()
agent2 = factory.create_formatted_agent("business")
agent3 = factory.create_formatted_agent("technical")
```

### 2. Dependency Injection

**Why:** Make testing easier and reduce coupling

```python
# Pass dependencies instead of creating them inside
def create_agent(llm, tools, prompt):
    agent = create_openai_functions_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools)
```

### 3. Configuration Centralization

**Why:** Single source of truth for settings

```python
# All settings in one place
settings = get_settings()
llm = ChatOpenAI(model=settings.model_name)
```

### 4. Lazy Loading

**Why:** Only load resources when needed

```python
@lru_cache()
def get_database(db_uri: str):
    # Database connection cached and reused
    return SQLDatabase.from_uri(db_uri)
```

### 5. Tool Registry

**Why:** Centralized tool management

```python
# Register all tools in one place
def get_all_tools(llm=None, db=None):
    tools = [base_tools, file_tools, api_tools]
    if llm and db:
        tools.extend(get_sql_tools(llm, db))
    return tools
```

---

## Best Practices

### File Organization

| File Type  | Responsibility                  | Max Lines | When to Split              |
| ---------- | ------------------------------- | --------- | -------------------------- |
| **Tools**  | Single tool or related group    | ~100      | Group by functionality     |
| **Agents** | One agent type                  | ~200      | Separate by specialization |
| **Config** | Settings/prompts for one domain | ~150      | Separate by category       |
| **Utils**  | Helper functions                | ~100      | Keep utilities focused     |

### Naming Conventions

```python
# Files: lowercase with underscores
base_tools.py
sql_agent.py
formatting_chain.py

# Classes: PascalCase
class AgentFactory
class SQLAgent
class Settings

# Functions: snake_case
def get_all_tools()
def create_base_agent()
def load_schema_context()

# Tools: snake_case (decorated functions)
@tool
def calculator(expression: str) -> str:
    pass
```

### Import Organization

```python
# Standard library
import json
from functools import lru_cache

# Third-party packages
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor

# Local imports
from config.settings import get_settings
from tools import get_all_tools
from utils.database import get_database
```

### Environment Variables

**.env:**

```env
OPENAI_API_KEY=sk-...
MODEL_NAME=gpt-4
TEMPERATURE=0.0
MAX_ITERATIONS=10
VERBOSE=true

DB_PATH=sqlite:///example.db
DB_TABLES=users,orders,products
```

**config/settings.py:**

```python
class Settings(BaseSettings):
    openai_api_key: str
    model_name: str = "gpt-4"

    class Config:
        env_file = ".env"
```

---

## Migration Path

### From Monolithic to Structured

**Step 1: Extract Configuration**

- Move API keys and settings to `.env`
- Create `config/settings.py`

**Step 2: Separate Tools**

- Create `tools/` directory
- Group tools by functionality
- Create tool factory function

**Step 3: Extract Prompts**

- Move prompts to `config/prompts.py`
- Create prompt factory functions

**Step 4: Create Agent Factory**

- Move agent creation to `agents/base_agent.py`
- Implement factory pattern

**Step 5: Add Utilities**

- Extract database helpers to `utils/`
- Add logging, caching

**Step 6: Update Main**

- Simplify `main.py` to use factories
- Remove duplicated code

---

## Testing Structure

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── test_tools.py            # Tool unit tests
├── test_agents.py           # Agent integration tests
├── test_chains.py           # Chain tests
└── test_utils.py            # Utility tests
```

**conftest.py:**

```python
import pytest
from config.settings import Settings

@pytest.fixture
def test_settings():
    return Settings(
        openai_api_key="test-key",
        model_name="gpt-3.5-turbo",
        verbose=False,
    )

@pytest.fixture
def mock_llm():
    from unittest.mock import MagicMock
    return MagicMock()
```

---

## Quick Reference

### When to Use Each Pattern

| Pattern                  | Use When                            | Example                  |
| ------------------------ | ----------------------------------- | ------------------------ |
| **Factory**              | Creating multiple agent variants    | Different output formats |
| **Dependency Injection** | Testing or swapping implementations | Mock LLM for testing     |
| **Lazy Loading**         | Expensive resources                 | Database connections     |
| **Configuration**        | Settings shared across modules      | API keys, model names    |
| **Tool Registry**        | Managing many tools                 | 10+ tools                |

### Common Mistakes to Avoid

❌ **Don't:** Put everything in one file  
✅ **Do:** Separate by concern (tools, agents, config)

❌ **Don't:** Hard-code settings in code  
✅ **Do:** Use environment variables and config files

❌ **Don't:** Create database connections everywhere  
✅ **Do:** Use cached factory function

❌ **Don't:** Duplicate agent creation code  
✅ **Do:** Use factory pattern

❌ **Don't:** Mix tool logic with tool definitions  
✅ **Do:** Separate tool wrappers from implementation

---

## Summary

**Key Principles:**

1. **Separation of Concerns** - Each file has one responsibility
2. **DRY (Don't Repeat Yourself)** - Use factories and utilities
3. **Configuration Management** - Centralized settings
4. **Testability** - Each module independently testable
5. **Maintainability** - Clear structure, easy to find code
6. **Scalability** - Easy to add new tools/agents

**Recommended Approach:**

1. Start with 3 directories: `config/`, `tools/`, `agents/`
2. Use factory pattern for agent creation
3. Centralize configuration in `.env` and `settings.py`
4. Group tools by functionality
5. Add `utils/` as needed for shared code

---

_Last Updated: January 25, 2026_
