# Output Formatting Strategies for LangChain Agents

Collection of approaches to control how an agent formats its final output based on context/category.

---

## Table of Contents

- [Output Formatting Strategies for LangChain Agents](#output-formatting-strategies-for-langchain-agents)
  - [Table of Contents](#table-of-contents)
  - [Chapter 1: Overview \& Comparison](#chapter-1-overview--comparison)
    - [All Approaches at a Glance](#all-approaches-at-a-glance)
    - [Approach 1: Simple Tool-Based Context Return](#approach-1-simple-tool-based-context-return)
    - [Approach 2: Structured Output with Pydantic Models](#approach-2-structured-output-with-pydantic-models)
    - [Approach 3: Dynamic System Prompt Injection](#approach-3-dynamic-system-prompt-injection)
    - [Approach 4: Post-Processing Chain](#approach-4-post-processing-chain)
    - [Approach 5: Using `with_structured_output()`](#approach-5-using-with_structured_output)
    - [Decision Matrix](#decision-matrix)
    - [When to Use Each Approach](#when-to-use-each-approach)
    - [Recommended Combinations](#recommended-combinations)
  - [Approach 1: Simple Tool-Based Context Return](#approach-1-simple-tool-based-context-return-1)
  - [Approach 2: Structured Output with Pydantic Models](#approach-2-structured-output-with-pydantic-models-1)
  - [Approach 3: Dynamic System Prompt Injection](#approach-3-dynamic-system-prompt-injection-1)
  - [Approach 4: Post-Processing Chain](#approach-4-post-processing-chain-1)
  - [Table of Contents](#table-of-contents-1)
- [Create a chain with post-processing:](#create-a-chain-with-post-processing)
- [Usage:](#usage)
  - [Comparison Matrix](#comparison-matrix)
  - [Recommendations](#recommendations)
  - [Example: Combined Approach](#example-combined-approach)

---

## Chapter 1: Overview & Comparison

### All Approaches at a Glance

This guide covers 5 distinct strategies for controlling output formatting in LangChain agents. Each approach has specific use cases, advantages, and trade-offs.

---

### Approach 1: Simple Tool-Based Context Return

**What It Does:** Creates a tool that returns formatting instructions as text, which the agent reads and applies to its final output.

**Use Cases:**

- Rapid prototyping with minimal code changes
- User-driven format selection at runtime
- Experimentation with different output styles
- Non-critical formatting where approximate adherence is acceptable

**Pros:**

- ✅ Minimal code complexity - just a simple function
- ✅ Easy to add to existing agents without refactoring
- ✅ Users can dynamically request different formats
- ✅ No additional LLM calls or token overhead
- ✅ Flexible - agent interprets instructions creatively

**Cons:**

- ❌ No guarantees - agent may ignore or misinterpret instructions
- ❌ Inconsistent results across different queries
- ❌ Agent must "remember" to apply formatting
- ❌ Hard to validate output structure
- ❌ Depends heavily on LLM's instruction-following capability

**Best For:** Quick implementations, experimentation, casual applications

---

### Approach 2: Structured Output with Pydantic Models

**What It Does:** Defines formatting configuration as typed Pydantic models that provide structured schema for format specifications.

**Use Cases:**

- Type-safe configuration management
- When you need to serialize/store format configurations
- API integrations requiring schema validation
- Configuration-driven applications

**Pros:**

- ✅ Type safety with Pydantic validation
- ✅ Clear, documented schema
- ✅ Serializable to JSON/YAML
- ✅ IDE autocomplete support
- ✅ Can validate format specifications before use
- ✅ Reusable across different parts of application

**Cons:**

- ❌ More boilerplate code to maintain
- ❌ Still relies on agent interpretation (no enforcement)
- ❌ Additional dependency (Pydantic)
- ❌ Requires JSON serialization/parsing
- ❌ Overkill for simple use cases

**Best For:** Production systems with configuration management, type-safe applications

---

### Approach 3: Dynamic System Prompt Injection

**What It Does:** Injects formatting instructions directly into the system prompt before agent execution begins.

**Use Cases:**

- Maximum control over formatting behavior
- When format is known before agent starts
- Critical applications requiring consistent output
- Setting organization-wide formatting standards

**Pros:**

- ✅ Highest level of control - agent sees instructions from start
- ✅ No tool call overhead
- ✅ Instructions present throughout entire conversation
- ✅ Can combine with other approaches
- ✅ Works with any LLM (no special features needed)
- ✅ Better adherence than tool-based approach

**Cons:**

- ❌ Requires prompt restructuring
- ❌ Format must be decided before execution
- ❌ Less dynamic - harder to change mid-conversation
- ❌ Increases token count in every message
- ❌ May conflict with other system instructions

**Best For:** Production applications, consistent formatting requirements, enterprise use

---

### Approach 4: Post-Processing Chain

**What It Does:** Applies formatting transformations to agent output after execution completes.

**Use Cases:**

- Retrofitting existing agents
- Applying consistent branding/styling
- Format conversion (e.g., Markdown to HTML)
- When agent output structure is predictable

**Pros:**

- ✅ Clean separation of concerns
- ✅ Can reformat existing outputs without re-running agent
- ✅ Easy to test formatting independently
- ✅ Multiple format variations from same output
- ✅ No impact on agent's reasoning process
- ✅ Works with any agent implementation

**Cons:**

- ❌ May lose context during transformation
- ❌ Limited to superficial formatting changes
- ❌ Can't influence content structure during generation
- ❌ Requires predictable output patterns
- ❌ Additional processing step adds latency

**Best For:** Existing systems, output standardization, multi-format support

---

### Approach 5: Using `with_structured_output()`

**What It Does:** Uses LLM's native structured output capability to enforce a specific schema on the final response.

**Use Cases:**

- API responses requiring exact schema
- Database insertions with strict typing
- Integration with downstream systems
- When output must be machine-parseable

**Pros:**

- ✅ Guaranteed output structure (enforced by LLM)
- ✅ Type-safe results with Pydantic
- ✅ No parsing errors or validation issues
- ✅ Perfect for API/database integration
- ✅ Works excellently with GPT-4 and newer models
- ✅ Eliminates output validation code

**Cons:**

- ❌ Requires additional LLM call (more tokens/cost)
- ❌ Adds latency (two-step process)
- ❌ Only works with models supporting structured output
- ❌ Less flexible - strict schema enforcement
- ❌ May truncate or omit information to fit schema
- ❌ Overkill for simple text formatting

**Best For:** API endpoints, database operations, strict schema requirements

---

### Decision Matrix

| Approach              | Complexity  | Control Level      | Flexibility | Token Cost  | Latency     | Expected Quality   | Schema Guarantee |
| --------------------- | ----------- | ------------------ | ----------- | ----------- | ----------- | ------------------ | ---------------- |
| **Tool-Based**        | ⭐ Low      | ⭐⭐ Medium        | ⭐⭐⭐ High | ⭐ Low      | ⭐ Low      | ⭐⭐ Variable      | ❌ No            |
| **Pydantic Models**   | ⭐⭐ Medium | ⭐⭐ Medium        | ⭐⭐⭐ High | ⭐ Low      | ⭐ Low      | ⭐⭐ Variable      | ❌ No            |
| **Dynamic Prompt**    | ⭐⭐ Medium | ⭐⭐⭐ High        | ⭐⭐ Medium | ⭐⭐ Medium | ⭐ Low      | ⭐⭐⭐ Good        | ❌ No            |
| **Post-Processing**   | ⭐ Low      | ⭐⭐ Medium        | ⭐⭐⭐ High | ⭐ Low      | ⭐⭐ Medium | ⭐⭐⭐ Good        | ⚠️ Partial       |
| **Structured Output** | ⭐⭐⭐ High | ⭐⭐⭐⭐ Very High | ⭐ Low      | ⭐⭐⭐ High | ⭐⭐⭐ High | ⭐⭐⭐⭐ Excellent | ✅ Yes           |

---

### When to Use Each Approach

**Choose Tool-Based (Approach 1) when:**

- You're prototyping or experimenting
- Users need to specify format preferences
- Approximate formatting is acceptable
- You want minimal code changes

**Choose Pydantic Models (Approach 2) when:**

- You need type-safe configuration
- Format specs are stored/loaded from files
- You're building a config-driven system
- IDE support and validation are important

**Choose Dynamic Prompt (Approach 3) when:**

- Formatting is critical and must be consistent
- Format is known at initialization time
- You need maximum control without extra LLM calls
- This is a production system with standards

**Choose Post-Processing (Approach 4) when:**

- Adding formatting to existing agents
- You need multiple output formats from same content
- Agent output follows predictable patterns
- You want clean separation between logic and presentation

**Choose Structured Output (Approach 5) when:**

- Output feeds into APIs or databases
- Schema validation is mandatory
- You can afford the extra token cost
- Using GPT-4 or compatible models
- Machine-readable output is required

---

### Recommended Combinations

**Best for Most Applications:**

- **Dynamic Prompt + Post-Processing**: Control during generation + flexibility after

**Best for Production Systems:**

- **Dynamic Prompt + Structured Output**: Strong guidance + guaranteed schema

**Best for User-Facing Apps:**

- **Tool-Based + Dynamic Prompt**: User choice + default formatting

**Best for Rapid Development:**

- **Tool-Based + Post-Processing**: Quick to implement + easy to iterate

---

## Approach 1: Simple Tool-Based Context Return

**Use Case:** Quick, simple formatting hints  
**Pros:** Easy to implement, agent decides how to apply  
**Cons:** Less control, depends on agent interpretation

```python
from langchain_core.tools import tool

@tool
def get_output_format(category: str) -> str:
    """Get formatting guidelines for the final output based on category.
    Categories: 'technical', 'business', 'casual', 'detailed', 'summary'"""

    formats = {
        "technical": """
Format the response as:
- Use technical terminology
- Include specific metrics and data points
- Structure: Overview → Details → Technical Notes
- Use bullet points for clarity
        """,
        "business": """
Format the response as:
- Executive summary at top
- Key metrics and KPIs
- Business impact analysis
- Recommendations section
- Professional tone
        """,
        "casual": """
Format the response as:
- Conversational, friendly tone
- Simple language, avoid jargon
- Use analogies where helpful
- Keep it concise and easy to read
        """,
        "detailed": """
Format the response as:
- Comprehensive breakdown of all data
- Step-by-step explanations
- Include raw numbers and calculations
- Context for each data point
        """,
        "summary": """
Format the response as:
- Brief overview (2-3 sentences max per topic)
- Only highlight key findings
- Use numbers/percentages
- Omit implementation details
        """,
    }

    return formats.get(
        category.lower(),
        "Standard format: Clear, concise, organized by topic."
    )

# Usage in agent:
tools = [
    calculator,
    weather,
    get_output_format,  # Add to tools list
    *sql_tools,
]

# Example question:
question = (
    "First, get the output format for 'business' category. "
    "Then tell me: How many users are registered? "
    "Format your final response according to the business category guidelines."
)
```

---

## Approach 2: Structured Output with Pydantic Models

**Use Case:** Type-safe, structured formatting configuration  
**Pros:** Type checking, clear schema, serializable  
**Cons:** More boilerplate, requires JSON handling

```python
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.tools import tool

class OutputFormat(BaseModel):
    """Define the structure and style for the output."""
    format_type: Literal["technical", "business", "casual", "detailed", "summary"]
    tone: str = Field(description="The tone to use in the response")
    structure: list[str] = Field(description="Sections to include in order")
    max_length: str = Field(description="Length guideline")
    include_examples: bool = Field(default=False, description="Whether to include examples")
    data_presentation: str = Field(description="How to present data (tables, lists, paragraphs)")

@tool
def get_output_format(category: str) -> dict:
    """Get structured formatting guidelines based on category.
    Returns a detailed format specification as a dictionary."""

    formats = {
        "business": {
            "format_type": "business",
            "tone": "professional, executive-focused, ROI-oriented",
            "structure": ["Executive Summary", "Key Metrics", "Analysis", "Recommendations"],
            "max_length": "concise, 3-4 paragraphs max",
            "include_examples": False,
            "data_presentation": "Use tables for metrics, bullet points for recommendations"
        },
        "technical": {
            "format_type": "technical",
            "tone": "precise, technical, detailed",
            "structure": ["Overview", "Technical Details", "Metrics", "Implementation Notes"],
            "max_length": "detailed as needed, comprehensive",
            "include_examples": True,
            "data_presentation": "Code blocks, tables with technical specs, diagrams if relevant"
        },
        "casual": {
            "format_type": "casual",
            "tone": "friendly, conversational, approachable",
            "structure": ["Quick Overview", "Main Points", "Bottom Line"],
            "max_length": "short and sweet, 2-3 paragraphs",
            "include_examples": True,
            "data_presentation": "Simple lists, avoid tables, use analogies"
        },
        "detailed": {
            "format_type": "detailed",
            "tone": "comprehensive, explanatory, educational",
            "structure": ["Introduction", "Step-by-Step Analysis", "Raw Data", "Context", "Conclusion"],
            "max_length": "as long as needed for completeness",
            "include_examples": True,
            "data_presentation": "Tables, code blocks, detailed breakdowns"
        },
        "summary": {
            "format_type": "summary",
            "tone": "brief, to-the-point, highlight-focused",
            "structure": ["Key Findings Only"],
            "max_length": "2-3 sentences per topic maximum",
            "include_examples": False,
            "data_presentation": "Only critical numbers, percentages, no tables"
        }
    }

    return formats.get(category.lower(), formats["business"])

# Usage:
format_spec = get_output_format.invoke("business")
print(f"Using format: {format_spec['format_type']}")
print(f"Tone: {format_spec['tone']}")
print(f"Structure: {', '.join(format_spec['structure'])}")
```

---

## Approach 3: Dynamic System Prompt Injection

**Use Case:** Maximum flexibility, format instructions in system prompt  
**Pros:** Most control, agent sees instructions from the start  
**Cons:** Requires prompt restructuring

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough

def get_format_instructions(category: str) -> str:
    """Return system-level instructions for output formatting."""
    instructions = {
        "business": """
FINAL OUTPUT FORMAT REQUIREMENTS:
1. Start with an Executive Summary (2-3 sentences)
2. Present Key Metrics in a table or bullet list
3. Provide Analysis section explaining the numbers
4. End with actionable Recommendations
5. Use professional, executive-focused language
6. Focus on ROI and business impact
        """,
        "technical": """
FINAL OUTPUT FORMAT REQUIREMENTS:
1. Begin with Technical Overview
2. Include all relevant metrics with units
3. Show code snippets or SQL queries used
4. Provide implementation notes
5. Use precise technical terminology
6. Format data in tables where appropriate
        """,
        "casual": """
FINAL OUTPUT FORMAT REQUIREMENTS:
1. Use friendly, conversational tone
2. Explain concepts with analogies
3. Keep it short and easy to read
4. Avoid technical jargon
5. Use simple bullet points
6. Add context so anyone can understand
        """,
        "detailed": """
FINAL OUTPUT FORMAT REQUIREMENTS:
1. Comprehensive breakdown of all findings
2. Show all calculations and raw data
3. Explain each step of the analysis
4. Include context for every data point
5. Use tables, lists, and detailed paragraphs
6. Don't skip any relevant information
        """,
        "summary": """
FINAL OUTPUT FORMAT REQUIREMENTS:
1. Only 2-3 sentences per topic
2. Highlight only key findings
3. Use percentages and critical numbers
4. Omit all implementation details
5. Be extremely concise
6. Focus on the "so what?"
        """
    }
    return instructions.get(category, instructions["business"])

# Create dynamic prompt:
def create_formatted_prompt(format_category: str = "business"):
    format_instruction = get_format_instructions(format_category)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant with access to various tools including a database.

{format_instruction}

Database Schema Context:
{schema_context}

When querying the database:
- Always check the schema first if unsure about table structure
- Use proper SQL syntax for SQLite
- Be careful with date formats and data types

Apply the formatting instruction ONLY to your FINAL response after gathering all data."""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    return prompt.partial(format_instruction=format_instruction)

# Usage:
prompt = create_formatted_prompt(format_category="business")
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

---

## Approach 4: Post-Processing Chain

**Use Case:** Apply formatting after agent completes  
**Pros:** Clean separation, can reformat existing outputs  
**Cons:** May lose some context

````python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

def format_output(text: str, category: str) -> str:
    """Apply formatting rules to the final output."""

    formatters = {
        "business": lambda t: f"""## Executive Summary

{t}

---
*Business Analysis Report*
""",
        "technical": lambda t: f"""```
TECHNICAL REPORT
================
````

{t}

---

Technical Details & Metrics
""",
"summary": lambda t: "\n".join([
line for line in t.split('\n')
if line.strip() and not line.startswith('##')
][:3]), # Only first 3 non-empty lines

        "casual": lambda t: f"Hey! Here's what I found:\n\n{t}\n\nHope that helps! 😊",

        "detailed": lambda t: f"""# Detailed Analysis Report

## Table of Contents

1. Overview
2. Detailed Findings
3. Raw Data
4. Conclusions

---

{t}

---

_Comprehensive Report - {category.title()} Format_
"""
}

    formatter = formatters.get(category, lambda t: t)
    return formatter(text)

# Create a chain with post-processing:

def create_formatting_chain(agent_executor, category: str = "business"):
"""Wrap agent executor with output formatting."""
return (
agent_executor
| RunnableLambda(lambda x: format_output(x["output"], category))
)

# Usage:

chain = create_formatting_chain(agent_executor, category="business")
response = chain.invoke({"input": question})
print(response) # Already formatted

````

---

## Approach 5: Using `with_structured_output()`

**Use Case:** Enforce specific output schema (GPT-4 best)
**Pros:** Guaranteed structure, type-safe output
**Cons:** Requires additional LLM call, more tokens

```python
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class BusinessReport(BaseModel):
    """Business-focused report format."""
    executive_summary: str
    key_metrics: list[str]
    detailed_findings: str
    recommendations: list[str]

class TechnicalReport(BaseModel):
    """Technical report format."""
    overview: str
    technical_details: str
    metrics: dict[str, str]
    implementation_notes: list[str]
    code_snippets: list[str] = []

class CasualResponse(BaseModel):
    """Casual, friendly response format."""
    quick_overview: str
    main_points: list[str]
    bottom_line: str

# Map categories to schemas:
REPORT_SCHEMAS = {
    "business": BusinessReport,
    "technical": TechnicalReport,
    "casual": CasualResponse,
}

def format_with_structure(agent_output: str, category: str = "business"):
    """Format agent output using structured output."""

    formatter_llm = ChatOpenAI(temperature=0, model="gpt-4")
    schema = REPORT_SCHEMAS.get(category, BusinessReport)

    structured_formatter = formatter_llm.with_structured_output(schema)

    prompt = f"""Format the following information as a {category} report:

{agent_output}

Extract and organize the information according to the required structure."""

    return structured_formatter.invoke(prompt)

# Usage:
response = agent_executor.invoke({"input": question})
formatted_report = format_with_structure(response["output"], category="business")

print(f"Executive Summary: {formatted_report.executive_summary}")
print(f"Key Metrics: {', '.join(formatted_report.key_metrics)}")
print(f"Recommendations: {formatted_report.recommendations}")

# Or convert to dict:
report_dict = formatted_report.model_dump()
````

---

## Comparison Matrix

| Approach          | Complexity | Control   | Flexibility | Token Cost | Best For             |
| ----------------- | ---------- | --------- | ----------- | ---------- | -------------------- |
| Tool-based        | Low        | Medium    | High        | Low        | Quick implementation |
| Pydantic Models   | Medium     | High      | Medium      | Low        | Type safety needed   |
| Dynamic Prompt    | Medium     | High      | High        | Low        | Most scenarios       |
| Post-Processing   | Low        | Medium    | High        | Low        | Existing systems     |
| Structured Output | High       | Very High | Low         | High       | Strict schemas       |

---

## Recommendations

1. **Start with Dynamic Prompt (Approach 3)** - Best balance of control and flexibility
2. **Add Tool-based (Approach 1)** - If users need to specify format dynamically
3. **Use Structured Output (Approach 5)** - When you need guaranteed output schema (e.g., API responses)
4. **Combine approaches** - Use dynamic prompt + post-processing for best results

---

## Example: Combined Approach

```python
# Best of all worlds:
def create_agent_with_formatting(
    llm,
    tools,
    default_format: str = "business",
    enable_structured_output: bool = False
):
    # 1. Add format tool to allow dynamic override
    @tool
    def get_output_format(category: str) -> str:
        """Get formatting guidelines."""
        return get_format_instructions(category)

    all_tools = [*tools, get_output_format]

    # 2. Create prompt with default formatting instruction
    format_instruction = get_format_instructions(default_format)
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""You are a helpful assistant.

{format_instruction}

{{schema_context}}"""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    agent = create_openai_functions_agent(llm, all_tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True)

    # 3. Optionally add post-processing or structured output
    if enable_structured_output:
        return (
            agent_executor
            | RunnableLambda(
                lambda x: format_with_structure(x["output"], default_format)
            )
        )

    return agent_executor

# Usage:
agent = create_agent_with_formatting(
    llm=llm,
    tools=tools,
    default_format="business",
    enable_structured_output=True
)
```

---

_Last Updated: January 25, 2026_
