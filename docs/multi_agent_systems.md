# Multi-Agent Systems Guide

Comprehensive guide to building systems with multiple specialized agents.

---

## Table of Contents

1. [Overview](#overview)
2. [Agent Patterns](#agent-patterns)
3. [Supervisor Pattern](#supervisor-pattern)
4. [Sequential Agents](#sequential-agents)
5. [Parallel Execution](#parallel-execution)
6. [Agent Communication](#agent-communication)
7. [LangGraph Integration](#langgraph-integration)
8. [Production Examples](#production-examples)

---

## Overview

### Why Multi-Agent Systems?

**Single Agent Limitations:**

- Too many tools (cognitive overload)
- Conflicting requirements
- Complex workflows
- Specialized expertise needed

**Multi-Agent Benefits:**

- ✅ Specialized agents for specific tasks
- ✅ Better performance on complex problems
- ✅ Easier maintenance and testing
- ✅ Parallel execution for speed
- ✅ Clear separation of concerns

**Use Cases:**

- Customer support (routing, resolution, escalation)
- Research (searching, analysis, summarization)
- Data pipeline (extraction, transformation, validation)
- E-commerce (recommendation, checkout, support)

---

## Agent Patterns

### Pattern 1: Single Responsibility Agents

```python
# agents/specialized_agents.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class DataExtractionAgent:
    """Agent specialized in extracting data from sources."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data extraction specialist.

Your ONLY job is to extract structured data from text.
- Extract key entities (names, dates, amounts)
- Identify relationships
- Output in JSON format
- Do NOT analyze or make recommendations"""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.tools = [extract_entities_tool, parse_dates_tool]

        agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools)

    def extract(self, text: str) -> dict:
        """Extract data from text."""
        result = self.executor.invoke({"input": text})
        return result


class DataAnalysisAgent:
    """Agent specialized in analyzing extracted data."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a data analysis specialist.

Your job is to analyze structured data and provide insights.
- Calculate statistics
- Identify trends and patterns
- Compare data points
- Generate summary insights
- Do NOT extract raw data (that's another agent's job)"""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.tools = [calculator_tool, statistics_tool, chart_tool]

        agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools)

    def analyze(self, data: dict) -> dict:
        """Analyze extracted data."""
        result = self.executor.invoke({"input": str(data)})
        return result


class ReportGenerationAgent:
    """Agent specialized in creating reports."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a report generation specialist.

Your job is to create polished, professional reports.
- Structure information clearly
- Use appropriate formatting
- Include executive summaries
- Create visualizations
- Do NOT perform analysis (use provided insights)"""),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])

        self.tools = [format_report_tool, generate_pdf_tool]

        agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.executor = AgentExecutor(agent=agent, tools=self.tools)

    def generate(self, analysis: dict) -> str:
        """Generate report from analysis."""
        result = self.executor.invoke({"input": str(analysis)})
        return result["output"]
```

---

## Supervisor Pattern

### Supervisor Agent Orchestration

```python
# agents/supervisor.py
from typing import List, Dict, Any
from enum import Enum

class AgentType(Enum):
    """Available agent types."""
    EXTRACTOR = "extractor"
    ANALYZER = "analyzer"
    REPORTER = "reporter"

class SupervisorAgent:
    """Supervisor that routes tasks to specialized agents."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        # Initialize specialized agents
        self.agents = {
            AgentType.EXTRACTOR: DataExtractionAgent(llm),
            AgentType.ANALYZER: DataAnalysisAgent(llm),
            AgentType.REPORTER: ReportGenerationAgent(llm),
        }

        # Supervisor's routing logic
        self.routing_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a supervisor agent that routes tasks to specialists.

Available agents:
- extractor: Extracts data from text
- analyzer: Analyzes structured data
- reporter: Generates formatted reports

Your job:
1. Understand the user's request
2. Decide which agent(s) to use
3. Determine the order of operations
4. Route tasks appropriately

Respond with JSON:
{{
    "agents": ["extractor", "analyzer"],
    "reasoning": "Need to extract data first, then analyze it"
}}"""),
            ("user", "{input}"),
        ])

    def route_task(self, user_input: str) -> Dict[str, Any]:
        """Determine which agents to use."""

        result = self.llm.invoke(
            self.routing_prompt.format_messages(input=user_input)
        )

        # Parse routing decision
        import json
        routing = json.loads(result.content)

        return routing

    def execute(self, user_input: str) -> Dict[str, Any]:
        """Execute task using appropriate agents."""

        # Get routing decision
        routing = self.route_task(user_input)

        logger.info(
            "Supervisor routing decision",
            agents=routing["agents"],
            reasoning=routing["reasoning"]
        )

        # Execute agents in sequence
        current_data = user_input
        results = {}

        for agent_name in routing["agents"]:
            agent_type = AgentType(agent_name)
            agent = self.agents[agent_type]

            logger.info(f"Executing {agent_name} agent")

            if agent_type == AgentType.EXTRACTOR:
                current_data = agent.extract(current_data)
                results["extraction"] = current_data

            elif agent_type == AgentType.ANALYZER:
                current_data = agent.analyze(current_data)
                results["analysis"] = current_data

            elif agent_type == AgentType.REPORTER:
                current_data = agent.generate(current_data)
                results["report"] = current_data

        return {
            "final_output": current_data,
            "intermediate_results": results,
            "routing": routing
        }

# Usage
llm = ChatOpenAI(model="gpt-4", temperature=0)
supervisor = SupervisorAgent(llm)

result = supervisor.execute(
    "Extract sales data from this report and create an analysis"
)

print(result["final_output"])
```

---

## Sequential Agents

### Pipeline Pattern

```python
# agents/pipeline.py
from typing import List, Callable, Any
from dataclasses import dataclass

@dataclass
class PipelineStage:
    """A stage in the agent pipeline."""
    name: str
    agent: Any
    transform: Callable[[Any], Any] = None

class AgentPipeline:
    """Sequential pipeline of agents."""

    def __init__(self, stages: List[PipelineStage]):
        self.stages = stages

    def execute(self, initial_input: Any) -> Dict[str, Any]:
        """Execute pipeline stages sequentially."""

        current_data = initial_input
        results = {}

        for stage in self.stages:
            logger.info(f"Pipeline stage: {stage.name}")

            # Execute agent
            stage_result = stage.agent.invoke({"input": str(current_data)})

            # Transform output if needed
            if stage.transform:
                current_data = stage.transform(stage_result)
            else:
                current_data = stage_result.get("output", stage_result)

            # Store intermediate result
            results[stage.name] = current_data

        return {
            "final_output": current_data,
            "intermediate_results": results
        }

# Usage
pipeline = AgentPipeline([
    PipelineStage(
        name="extraction",
        agent=extractor_agent,
        transform=lambda r: r.get("output")
    ),
    PipelineStage(
        name="validation",
        agent=validator_agent,
        transform=lambda r: r.get("validated_data")
    ),
    PipelineStage(
        name="analysis",
        agent=analyzer_agent
    ),
    PipelineStage(
        name="reporting",
        agent=reporter_agent
    )
])

result = pipeline.execute("Process this sales data...")
print(result["final_output"])
```

---

## Parallel Execution

### Concurrent Agent Execution

```python
# agents/parallel.py
import asyncio
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

class ParallelAgentExecutor:
    """Execute multiple agents in parallel."""

    def __init__(self, agents: Dict[str, Any]):
        self.agents = agents

    def execute_parallel(
        self,
        tasks: Dict[str, str],
        max_workers: int = 3
    ) -> Dict[str, Any]:
        """
        Execute tasks in parallel.

        Args:
            tasks: {"agent_name": "task_input"}
            max_workers: Maximum parallel executions
        """

        results = {}

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_agent = {
                executor.submit(
                    self.agents[agent_name].invoke,
                    {"input": task_input}
                ): agent_name
                for agent_name, task_input in tasks.items()
            }

            # Collect results as they complete
            for future in as_completed(future_to_agent):
                agent_name = future_to_agent[future]

                try:
                    result = future.result()
                    results[agent_name] = result
                    logger.info(f"Agent {agent_name} completed")

                except Exception as e:
                    logger.error(f"Agent {agent_name} failed: {e}")
                    results[agent_name] = {"error": str(e)}

        return results

    async def execute_parallel_async(
        self,
        tasks: Dict[str, str]
    ) -> Dict[str, Any]:
        """Async version of parallel execution."""

        async def run_agent(agent_name: str, task_input: str):
            """Run single agent."""
            try:
                # Wrap sync agent in async
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.agents[agent_name].invoke,
                    {"input": task_input}
                )
                return agent_name, result

            except Exception as e:
                logger.error(f"Agent {agent_name} failed: {e}")
                return agent_name, {"error": str(e)}

        # Execute all agents concurrently
        tasks_list = [
            run_agent(agent_name, task_input)
            for agent_name, task_input in tasks.items()
        ]

        results_list = await asyncio.gather(*tasks_list)

        # Convert to dict
        results = {agent_name: result for agent_name, result in results_list}

        return results

# Usage - Research Multiple Topics in Parallel
research_agents = {
    "market_research": MarketResearchAgent(llm),
    "competitor_research": CompetitorResearchAgent(llm),
    "customer_research": CustomerResearchAgent(llm),
}

parallel_executor = ParallelAgentExecutor(research_agents)

results = parallel_executor.execute_parallel({
    "market_research": "Analyze AI market trends",
    "competitor_research": "Research top 3 competitors",
    "customer_research": "Survey customer needs"
})

# All research completed in parallel!
print(results)
```

---

## Agent Communication

### Message Passing Between Agents

```python
# agents/communication.py
from dataclasses import dataclass
from typing import Any, Optional
from datetime import datetime
from queue import Queue

@dataclass
class AgentMessage:
    """Message passed between agents."""
    from_agent: str
    to_agent: str
    message_type: str  # 'request', 'response', 'notification'
    content: Any
    timestamp: datetime = None
    correlation_id: str = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()

class MessageBus:
    """Central message bus for agent communication."""

    def __init__(self):
        self.queues: Dict[str, Queue] = {}
        self.message_history: List[AgentMessage] = []

    def register_agent(self, agent_name: str):
        """Register an agent with the message bus."""
        if agent_name not in self.queues:
            self.queues[agent_name] = Queue()

    def send_message(self, message: AgentMessage):
        """Send message to an agent."""

        # Store in history
        self.message_history.append(message)

        # Deliver to recipient's queue
        if message.to_agent in self.queues:
            self.queues[message.to_agent].put(message)
            logger.info(
                "Message sent",
                from_agent=message.from_agent,
                to_agent=message.to_agent,
                type=message.message_type
            )
        else:
            logger.error(f"Unknown agent: {message.to_agent}")

    def receive_message(
        self,
        agent_name: str,
        timeout: Optional[float] = None
    ) -> Optional[AgentMessage]:
        """Receive message for an agent."""

        if agent_name not in self.queues:
            return None

        try:
            message = self.queues[agent_name].get(timeout=timeout)
            return message
        except:
            return None

# Communicating Agents
class CollaborativeAgent:
    """Agent that can communicate with others."""

    def __init__(
        self,
        name: str,
        agent_executor: AgentExecutor,
        message_bus: MessageBus
    ):
        self.name = name
        self.agent_executor = agent_executor
        self.message_bus = message_bus

        # Register with message bus
        message_bus.register_agent(name)

    def send_request(self, to_agent: str, request: str) -> str:
        """Send request to another agent."""

        correlation_id = str(uuid.uuid4())

        # Send request
        message = AgentMessage(
            from_agent=self.name,
            to_agent=to_agent,
            message_type="request",
            content=request,
            correlation_id=correlation_id
        )

        self.message_bus.send_message(message)

        # Wait for response
        while True:
            response = self.message_bus.receive_message(self.name, timeout=30)

            if response and response.correlation_id == correlation_id:
                return response.content

            if response is None:
                raise TimeoutError(f"No response from {to_agent}")

    def process_message(self, message: AgentMessage) -> AgentMessage:
        """Process incoming message."""

        # Use agent to process request
        result = self.agent_executor.invoke({"input": message.content})

        # Send response
        response = AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="response",
            content=result["output"],
            correlation_id=message.correlation_id
        )

        return response

    def run(self):
        """Run agent message processing loop."""

        while True:
            # Check for messages
            message = self.message_bus.receive_message(self.name, timeout=1)

            if message:
                logger.info(f"{self.name} received message")

                # Process and respond
                response = self.process_message(message)
                self.message_bus.send_message(response)

# Usage
message_bus = MessageBus()

research_agent = CollaborativeAgent(
    name="researcher",
    agent_executor=research_agent_executor,
    message_bus=message_bus
)

writing_agent = CollaborativeAgent(
    name="writer",
    agent_executor=writing_agent_executor,
    message_bus=message_bus
)

# Researcher asks writer for help
response = research_agent.send_request(
    to_agent="writer",
    request="Summarize this research: ..."
)
```

---

## LangGraph Integration

### State-Based Multi-Agent Workflow

```python
# agents/langgraph_workflow.py
from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage

class AgentState(TypedDict):
    """Shared state between agents."""
    messages: Annotated[list, "The messages in the conversation"]
    current_agent: str
    extracted_data: dict
    analysis_results: dict
    final_report: str
    next_step: str

def create_multi_agent_graph():
    """Create LangGraph workflow with multiple agents."""

    workflow = StateGraph(AgentState)

    # Define agent nodes
    def extraction_node(state: AgentState) -> AgentState:
        """Extract data from input."""
        extractor = DataExtractionAgent(llm)

        messages = state["messages"]
        last_message = messages[-1]

        result = extractor.extract(last_message.content)

        return {
            **state,
            "extracted_data": result,
            "current_agent": "extractor",
            "next_step": "analyze"
        }

    def analysis_node(state: AgentState) -> AgentState:
        """Analyze extracted data."""
        analyzer = DataAnalysisAgent(llm)

        result = analyzer.analyze(state["extracted_data"])

        return {
            **state,
            "analysis_results": result,
            "current_agent": "analyzer",
            "next_step": "report"
        }

    def reporting_node(state: AgentState) -> AgentState:
        """Generate final report."""
        reporter = ReportGenerationAgent(llm)

        result = reporter.generate(state["analysis_results"])

        return {
            **state,
            "final_report": result,
            "current_agent": "reporter",
            "next_step": "end"
        }

    # Add nodes
    workflow.add_node("extract", extraction_node)
    workflow.add_node("analyze", analysis_node)
    workflow.add_node("report", reporting_node)

    # Define routing logic
    def route_next(state: AgentState) -> str:
        """Determine next agent."""
        next_step = state.get("next_step", "extract")

        if next_step == "end":
            return END

        return next_step

    # Add edges
    workflow.set_entry_point("extract")
    workflow.add_conditional_edges(
        "extract",
        route_next,
        {"analyze": "analyze", END: END}
    )
    workflow.add_conditional_edges(
        "analyze",
        route_next,
        {"report": "report", END: END}
    )
    workflow.add_conditional_edges(
        "report",
        route_next,
        {END: END}
    )

    return workflow.compile()

# Usage
graph = create_multi_agent_graph()

initial_state = {
    "messages": [HumanMessage(content="Analyze this sales data: ...")],
    "current_agent": "start",
    "extracted_data": {},
    "analysis_results": {},
    "final_report": "",
    "next_step": "extract"
}

result = graph.invoke(initial_state)

print(result["final_report"])
```

---

## Production Examples

### Customer Support Multi-Agent System

```python
# agents/customer_support_system.py

class CustomerSupportSystem:
    """Multi-agent customer support system."""

    def __init__(self, llm: ChatOpenAI):
        self.llm = llm

        # Routing agent
        self.router = RoutingAgent(llm)

        # Specialized agents
        self.agents = {
            "order_status": OrderStatusAgent(llm),
            "returns": ReturnsAgent(llm),
            "technical": TechnicalSupportAgent(llm),
            "billing": BillingAgent(llm),
            "general": GeneralInquiryAgent(llm)
        }

        # Escalation agent
        self.escalation = EscalationAgent(llm)

    def handle_inquiry(self, user_input: str, user_id: str) -> dict:
        """Handle customer inquiry."""

        # Step 1: Route to appropriate agent
        routing = self.router.route(user_input)
        agent_type = routing["agent"]
        confidence = routing["confidence"]

        logger.info(
            "Inquiry routed",
            agent=agent_type,
            confidence=confidence
        )

        # Step 2: Execute specialized agent
        agent = self.agents[agent_type]
        result = agent.handle(user_input, user_id)

        # Step 3: Check if escalation needed
        if result.get("needs_escalation"):
            logger.info("Escalating to human agent")
            escalation_result = self.escalation.escalate(
                user_input=user_input,
                agent_result=result,
                user_id=user_id
            )
            return escalation_result

        return result

# Usage
support_system = CustomerSupportSystem(llm)

response = support_system.handle_inquiry(
    user_input="Where is my order #12345?",
    user_id="user_789"
)

print(response["message"])
```

---

## Best Practices

### 1. Keep Agents Focused

```python
# DON'T: One agent does everything
# DO: Specialized agents for specific tasks
extractor_agent  # Only extracts data
analyzer_agent   # Only analyzes data
reporter_agent   # Only generates reports
```

### 2. Define Clear Interfaces

```python
class AgentInterface:
    def process(self, input: dict) -> dict:
        """Standard interface for all agents."""
        pass
```

### 3. Handle Agent Failures

```python
try:
    result = agent.invoke(input)
except Exception as e:
    # Fallback to default agent or error handling
    result = fallback_agent.invoke(input)
```

### 4. Log Agent Decisions

```python
logger.info(
    "Agent decision",
    agent=agent_name,
    routing=routing_decision,
    confidence=confidence_score
)
```

### 5. Use Async for Parallel Execution

```python
async def execute_agents():
    results = await asyncio.gather(
        agent1.invoke_async(input1),
        agent2.invoke_async(input2),
        agent3.invoke_async(input3)
    )
```

---

_Last Updated: January 25, 2026_
