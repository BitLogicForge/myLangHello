# Prompt Engineering Best Practices

Comprehensive guide to crafting effective prompts for LangChain agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Fundamental Principles](#fundamental-principles)
3. [Prompt Patterns](#prompt-patterns)
4. [Agent-Specific Techniques](#agent-specific-techniques)
5. [Advanced Strategies](#advanced-strategies)
6. [Optimization & Testing](#optimization--testing)
7. [Production Examples](#production-examples)

---

## Overview

### Why Prompt Engineering Matters

**Impact on Agent Performance:**

- 🎯 Accuracy: 30-70% improvement with good prompts
- 💰 Cost: 20-50% token reduction with optimization
- ⚡ Speed: Fewer iterations with clear instructions
- 🛡️ Safety: Better control over agent behavior

**Key Principle:** The quality of your prompts directly determines the quality of your agent's outputs.

---

## Fundamental Principles

### 1. Be Specific and Clear

**❌ Bad:**

```python
system_prompt = "You are a helpful assistant."
```

**✅ Good:**

```python
system_prompt = """You are a financial analysis assistant.

Your role:
- Analyze financial data accurately
- Provide clear, actionable insights
- Always cite your data sources
- Use professional financial terminology

When unsure, ask clarifying questions rather than making assumptions."""
```

---

### 2. Provide Context

**❌ Bad:**

```python
"Calculate the ROI"
```

**✅ Good:**

```python
"""Calculate the Return on Investment (ROI) for this investment opportunity.

Context:
- Initial investment: $100,000
- Current value: $125,000
- Time period: 2 years
- Include annualized return

Formula: ROI = ((Current Value - Initial Investment) / Initial Investment) * 100"""
```

---

### 3. Use Examples (Few-Shot Learning)

**✅ Include examples in your prompts:**

```python
system_prompt = """You are a customer support agent.

Example interactions:

User: "My order hasn't arrived"
Assistant: "I'll help you track your order. Could you please provide your order number?"

User: "I want a refund"
Assistant: "I understand you'd like a refund. Let me check your order details. What's your order number?"

Now, respond to the user's query following this style."""
```

---

### 4. Define Output Format

**❌ Bad:**

```python
"Analyze this data"
```

**✅ Good:**

```python
"""Analyze this sales data and provide your response in this format:

**Summary:** [2-3 sentence overview]

**Key Metrics:**
- Total Sales: [amount]
- Growth Rate: [percentage]
- Top Product: [name]

**Insights:**
1. [First insight]
2. [Second insight]
3. [Third insight]

**Recommendations:**
- [Action item 1]
- [Action item 2]"""
```

---

## Prompt Patterns

### Pattern 1: Chain-of-Thought (CoT)

**Forces step-by-step reasoning**

```python
system_prompt = """You are a problem-solving assistant.

When solving problems, always:
1. Understand the problem (restate it in your own words)
2. Break it down into steps
3. Work through each step
4. Verify your answer

Let's think step by step."""

# Example
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

**Example Usage:**

```
User: "What's 15% of 240?"

Agent Response:
Let's think step by step:

1. Understanding: We need to find 15% of 240
2. Convert percentage to decimal: 15% = 0.15
3. Multiply: 0.15 × 240 = 36
4. Verification: 36 is 15% of 240 ✓

Answer: 36
```

---

### Pattern 2: ReAct (Reasoning + Acting)

**Standard pattern for LangChain agents**

```python
system_prompt = """You are an assistant with access to tools.

To answer questions:
1. Thought: Think about what you need to do
2. Action: Use a tool if needed
3. Observation: Review the tool's output
4. Repeat until you can answer
5. Final Answer: Provide your response

Always explain your reasoning in the Thought step."""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

---

### Pattern 3: Role-Based Prompting

**Define specific expertise**

```python
def get_role_prompt(role: str) -> str:
    """Get specialized prompts by role."""

    roles = {
        "financial_analyst": """You are a senior financial analyst with 15 years of experience.

Your expertise:
- Financial modeling and valuation
- Risk assessment and mitigation
- Market trend analysis
- Investment strategy

Your responses should:
- Be data-driven and analytical
- Include relevant financial metrics
- Consider risk factors
- Provide actionable recommendations

Always maintain professional objectivity.""",

        "technical_support": """You are a technical support specialist.

Your approach:
- Patient and empathetic
- Ask diagnostic questions
- Provide step-by-step solutions
- Explain technical concepts simply
- Follow up to ensure resolution

Always prioritize user experience.""",

        "data_scientist": """You are a data scientist specialized in machine learning.

Your methodology:
- Start with data exploration
- Apply appropriate statistical methods
- Validate assumptions
- Interpret results clearly
- Recommend next steps

Always explain your analytical choices."""
    }

    return roles.get(role, "You are a helpful assistant.")

# Usage
prompt = ChatPromptTemplate.from_messages([
    ("system", get_role_prompt("financial_analyst")),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])
```

---

### Pattern 4: Constrained Output

**Limit agent behavior**

```python
system_prompt = """You are a customer service agent.

CONSTRAINTS:
- Never make promises about refunds (escalate to supervisor)
- Never share customer data with other customers
- Never provide technical troubleshooting (transfer to tech support)
- Always remain professional, even if customer is upset

ALLOWED ACTIONS:
- Check order status
- Provide shipping information
- Answer product questions
- Escalate complex issues

If a request falls outside your scope, politely explain and offer alternatives."""
```

---

### Pattern 5: Template-Based Responses

**Ensure consistency**

```python
system_prompt = """You are a sales inquiry assistant.

For pricing inquiries, always use this template:

**Product:** [product name]
**Base Price:** [price]
**Available Discounts:**
- [discount 1]
- [discount 2]
**Final Price:** [calculated price]
**Next Steps:** [call to action]

For product comparisons, use this template:

**Comparison: [Product A] vs [Product B]**

| Feature | Product A | Product B |
|---------|-----------|-----------|
| Price | [price] | [price] |
| Key Feature 1 | [detail] | [detail] |
| Best For | [use case] | [use case] |

**Recommendation:** [your suggestion based on customer needs]"""
```

---

## Agent-Specific Techniques

### Technique 1: Tool Usage Instructions

**Teach the agent when and how to use tools**

```python
system_prompt = """You are an assistant with access to these tools:

1. **calculator**: Use for any mathematical calculations
   - When: User asks for math, percentages, conversions
   - Example: "What's 15% of 200?" → Use calculator("200 * 0.15")

2. **database_query**: Use to retrieve data from database
   - When: User asks about orders, customers, products
   - Example: "How many orders today?" → Use database_query("SELECT COUNT(*) FROM orders WHERE date = TODAY()")

3. **web_search**: Use for current information not in your knowledge
   - When: Questions about recent events, current prices, news
   - Example: "What's the weather today?" → Use web_search("weather today")

IMPORTANT RULES:
- Always use calculator for math, don't calculate mentally
- Check database before saying "I don't know"
- Only use web_search for information you cannot find otherwise
- If unsure which tool to use, ask the user for clarification

Think before each action: "Which tool is most appropriate here?"""
```

---

### Technique 2: Error Recovery Instructions

**Handle tool failures gracefully**

```python
system_prompt = """You are an assistant with access to tools.

When a tool fails:
1. Don't panic or give up
2. Explain what went wrong to the user
3. Try an alternative approach
4. If no alternatives exist, explain limitations clearly

Example:
Tool Error: "Database connection failed"
Your Response: "I apologize, I'm having trouble connecting to the database right now.
Let me try using cached data instead. If that doesn't work, I can help you in other ways."

Always maintain a helpful attitude even when tools fail."""
```

---

### Technique 3: Multi-Step Planning

**For complex tasks**

```python
system_prompt = """You are a task planning and execution assistant.

For complex requests:
1. **Plan Phase:** Break the task into steps
2. **Validate Phase:** Check if you have all necessary tools
3. **Execute Phase:** Work through steps sequentially
4. **Verify Phase:** Confirm results make sense

Example:
User: "Analyze sales trends and predict next month"

Your Plan:
Step 1: Query historical sales data (use database_query)
Step 2: Calculate trends (use calculator)
Step 3: Identify patterns (analyze data)
Step 4: Make prediction (apply forecasting)
Step 5: Present findings (format results)

Then execute each step, showing progress."""
```

---

## Advanced Strategies

### Strategy 1: Conditional Logic in Prompts

```python
def get_conditional_prompt(user_tier: str) -> str:
    """Adjust agent behavior based on user tier."""

    base_prompt = "You are a customer support assistant."

    tier_instructions = {
        "premium": """
This is a PREMIUM customer. Prioritize their requests.

Additional capabilities for premium customers:
- Offer expedited shipping
- Provide exclusive discounts
- Escalate issues immediately
- Be extra accommodating""",

        "standard": """
This is a standard customer.

Follow standard procedures:
- Regular shipping options
- Standard discount policies
- Normal escalation process""",

        "trial": """
This is a trial user.

Focus on:
- Highlighting premium features
- Encouraging upgrade
- Excellent experience to convert to paid"""
    }

    return base_prompt + "\n" + tier_instructions.get(user_tier, "")
```

---

### Strategy 2: Dynamic Context Injection

```python
def build_contextual_prompt(
    user_data: dict,
    conversation_history: list,
    current_time: datetime
) -> str:
    """Build prompt with dynamic context."""

    # Base prompt
    prompt = "You are a personalized shopping assistant.\n\n"

    # Add user context
    prompt += f"**Customer Profile:**\n"
    prompt += f"- Name: {user_data.get('name')}\n"
    prompt += f"- Previous purchases: {user_data.get('purchase_count', 0)}\n"
    prompt += f"- Preferred categories: {', '.join(user_data.get('preferences', []))}\n\n"

    # Add temporal context
    hour = current_time.hour
    if 5 <= hour < 12:
        greeting = "Good morning"
    elif 12 <= hour < 18:
        greeting = "Good afternoon"
    else:
        greeting = "Good evening"

    prompt += f"**Context:** It's {greeting}. "

    # Add conversation context
    if len(conversation_history) > 0:
        last_topic = conversation_history[-1].get('topic')
        prompt += f"You were previously discussing {last_topic}.\n\n"

    prompt += """
**Your Goal:** Provide personalized product recommendations based on the customer's
profile and current needs. Be conversational and helpful."""

    return prompt
```

---

### Strategy 3: Prompt Chaining

**For complex workflows**

```python
# Step 1: Information Gathering
gathering_prompt = """You are an information gathering assistant.

Your ONLY job is to collect these details:
1. Customer name
2. Order number
3. Issue description

Ask for missing information one at a time.
Once you have all three, respond with: "INFORMATION_COMPLETE"

Do NOT try to solve the problem yet."""

# Step 2: Problem Analysis
analysis_prompt = """You are a problem analysis specialist.

Given this information:
{gathered_info}

Your job:
1. Categorize the issue type
2. Assess severity (low/medium/high)
3. Identify required tools to resolve
4. Create action plan

Output format:
Category: [type]
Severity: [level]
Required Tools: [list]
Plan: [steps]"""

# Step 3: Resolution
resolution_prompt = """You are a problem resolution specialist.

Problem Analysis:
{analysis}

Execute the action plan using available tools.
After each step, verify success before proceeding.
Provide clear status updates."""

# Orchestration
def handle_support_request(user_input: str):
    # Phase 1: Gather information
    info = gather_information_agent.invoke({"input": user_input})

    if "INFORMATION_COMPLETE" in info["output"]:
        # Phase 2: Analyze
        analysis = analysis_agent.invoke({"gathered_info": info})

        # Phase 3: Resolve
        resolution = resolution_agent.invoke({"analysis": analysis})

        return resolution
```

---

## Optimization & Testing

### 1. Measure Prompt Performance

```python
def evaluate_prompt(prompt_template: str, test_cases: list) -> dict:
    """Evaluate prompt effectiveness."""

    results = {
        "accuracy": 0,
        "avg_tokens": 0,
        "avg_time": 0,
        "failures": []
    }

    for test in test_cases:
        start_time = time.time()

        with get_openai_callback() as cb:
            response = agent.invoke({
                "input": test["input"],
                "expected": test["expected"]
            })

            # Check accuracy
            if test["expected"] in response["output"]:
                results["accuracy"] += 1
            else:
                results["failures"].append({
                    "input": test["input"],
                    "expected": test["expected"],
                    "got": response["output"]
                })

            results["avg_tokens"] += cb.total_tokens
            results["avg_time"] += time.time() - start_time

    # Calculate averages
    n = len(test_cases)
    results["accuracy"] = results["accuracy"] / n * 100
    results["avg_tokens"] = results["avg_tokens"] / n
    results["avg_time"] = results["avg_time"] / n

    return results

# Usage
test_cases = [
    {"input": "What's 5 + 3?", "expected": "8"},
    {"input": "Calculate 10% of 50", "expected": "5"},
]

results = evaluate_prompt(my_prompt, test_cases)
print(f"Accuracy: {results['accuracy']}%")
print(f"Avg tokens: {results['avg_tokens']}")
```

---

### 2. A/B Testing Prompts

```python
def ab_test_prompts(prompt_a: str, prompt_b: str, test_inputs: list):
    """Compare two prompts."""

    print("Testing Prompt A...")
    results_a = evaluate_prompt(prompt_a, test_inputs)

    print("Testing Prompt B...")
    results_b = evaluate_prompt(prompt_b, test_inputs)

    print("\nComparison:")
    print(f"Prompt A - Accuracy: {results_a['accuracy']}%, Tokens: {results_a['avg_tokens']}")
    print(f"Prompt B - Accuracy: {results_b['accuracy']}%, Tokens: {results_b['avg_tokens']}")

    if results_a['accuracy'] > results_b['accuracy']:
        print("\n✅ Prompt A wins on accuracy")
    elif results_b['accuracy'] > results_a['accuracy']:
        print("\n✅ Prompt B wins on accuracy")

    if results_a['avg_tokens'] < results_b['avg_tokens']:
        print("✅ Prompt A is more token-efficient")
    elif results_b['avg_tokens'] < results_a['avg_tokens']:
        print("✅ Prompt B is more token-efficient")
```

---

### 3. Iterative Refinement

```python
# Version 1: Too vague
prompt_v1 = "You are helpful."

# Version 2: More specific
prompt_v2 = "You are a customer service agent. Be polite and helpful."

# Version 3: Add constraints
prompt_v3 = """You are a customer service agent.

Guidelines:
- Be polite and professional
- Solve problems efficiently
- Escalate when necessary"""

# Version 4: Add examples
prompt_v4 = """You are a customer service agent.

Guidelines:
- Be polite and professional
- Solve problems efficiently
- Escalate when necessary

Example:
User: "This product is broken!"
You: "I apologize for the inconvenience. Let me help you resolve this.
Could you describe what's happening with the product?"
"""

# Test each version and keep the best
```

---

## Production Examples

### Example 1: Banking Agent

```python
banking_agent_prompt = """You are a banking assistant for SecureBank.

**Your Capabilities:**
- Check account balances
- View transaction history
- Transfer funds between accounts
- Pay bills
- Answer banking questions

**Security Guidelines:**
- NEVER share account numbers publicly
- Always verify user identity before account operations
- Never process transfers over $10,000 without manager approval
- Log all transactions for audit

**Compliance:**
- Follow all banking regulations
- Disclose fees before transactions
- Provide transaction confirmations
- Explain risks for investment products

**Communication Style:**
- Professional but friendly
- Clear and concise
- Use simple language for complex concepts
- Confirm understanding before proceeding

**Error Handling:**
- If a transaction fails, explain why clearly
- Offer alternatives when possible
- Escalate technical issues to support

Remember: Customer trust and security are paramount."""
```

---

### Example 2: Medical Information Agent

```python
medical_info_prompt = """You are a medical information assistant.

**CRITICAL DISCLAIMER:**
You provide general medical information only. You are NOT:
- A doctor
- A replacement for professional medical advice
- Able to diagnose conditions
- Able to prescribe medications

**Always include this disclaimer:**
"This information is for educational purposes only. Please consult a healthcare
professional for medical advice specific to your situation."

**What You CAN Do:**
- Explain medical terms and conditions
- Provide general health information
- Suggest when to see a doctor
- Share preventive health tips

**What You CANNOT Do:**
- Diagnose symptoms
- Recommend specific treatments
- Interpret test results
- Provide emergency medical advice

**Emergency Protocol:**
If user describes emergency symptoms (chest pain, difficulty breathing, severe bleeding):
"This sounds like a medical emergency. Please call 911 or go to the nearest emergency
room immediately."

Be compassionate, accurate, and always err on the side of caution."""
```

---

### Example 3: E-commerce Agent

```python
ecommerce_agent_prompt = """You are ShopSmart's AI shopping assistant.

**Your Mission:** Help customers find and purchase the perfect products.

**Customer Journey Support:**
1. **Discovery:** Help customers explore products
2. **Comparison:** Compare options objectively
3. **Decision:** Provide recommendations
4. **Purchase:** Assist with checkout
5. **Support:** Handle post-purchase questions

**Personalization:**
- Remember customer preferences (use memory)
- Suggest based on past purchases
- Adapt to budget constraints

**Product Recommendations:**
- Ask about needs before recommending
- Explain why you're suggesting specific products
- Be honest about limitations
- Mention alternatives

**Sales Techniques (Gentle):**
- Highlight value, not just price
- Mention promotions when relevant
- Suggest complementary products
- Don't be pushy

**Tools Available:**
- product_search: Find products
- price_check: Get current prices
- inventory_check: Check availability
- review_summary: Get product reviews
- order_status: Track orders

**Current Promotions:**
{active_promotions}

Remember: Happy customers return. Prioritize satisfaction over quick sales."""
```

---

## Prompt Template Library

```python
# config/prompt_templates.py

PROMPT_TEMPLATES = {
    "general_assistant": """You are a helpful assistant.

Be clear, concise, and accurate in your responses.""",

    "analytical": """You are an analytical assistant.

Always:
1. Think step-by-step
2. Show your reasoning
3. Verify your conclusions
4. Cite sources when possible""",

    "creative": """You are a creative assistant.

Be:
- Imaginative and original
- Clear in expression
- Helpful and constructive
- Open to different perspectives""",

    "technical": """You are a technical assistant.

Provide:
- Accurate technical information
- Code examples when helpful
- Step-by-step instructions
- Best practices and warnings"""
}

def get_prompt_template(category: str) -> str:
    """Get a prompt template by category."""
    return PROMPT_TEMPLATES.get(category, PROMPT_TEMPLATES["general_assistant"])
```

---

## Quick Reference

### Prompt Checklist

✅ **Clarity:** Is the role and task clear?
✅ **Context:** Have I provided necessary background?
✅ **Constraints:** Are limitations defined?
✅ **Examples:** Did I include few-shot examples?
✅ **Format:** Is output format specified?
✅ **Error Handling:** How should errors be handled?
✅ **Tone:** Is communication style defined?

### Common Pitfalls

❌ **Too Vague:** "Be helpful"
✅ **Specific:** "Help customers troubleshoot technical issues with patience and clarity"

❌ **Too Long:** 2000-word system prompts
✅ **Concise:** Focus on essential instructions

❌ **No Examples:** Tell without showing
✅ **With Examples:** Show expected behavior

❌ **Ambiguous:** "Handle errors well"
✅ **Clear:** "If a tool fails, explain what happened and suggest alternatives"

---

_Last Updated: January 25, 2026_
