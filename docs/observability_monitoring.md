# Observability & Monitoring Guide

Comprehensive guide to monitoring and debugging LangChain agents in production.

---

## Table of Contents

1. [Overview](#overview)
2. [Logging Strategies](#logging-strategies)
3. [LangSmith Integration](#langsmith-integration)
4. [Custom Tracing](#custom-tracing)
5. [Metrics & Analytics](#metrics--analytics)
6. [Alerting](#alerting)
7. [Production Dashboards](#production-dashboards)

---

## Overview

### Why Observability Matters

**Production Challenges:**

- Agent makes wrong decisions
- Tools fail silently
- Costs spiral out of control
- Users report issues you can't reproduce
- Performance degrades over time

**Observability Solutions:**

- 📊 **Metrics:** Track performance, costs, errors
- 📝 **Logging:** Record every action and decision
- 🔍 **Tracing:** Follow request through entire system
- 🚨 **Alerting:** Get notified of issues immediately

---

## Logging Strategies

### Basic Logging Setup

```python
# utils/logger.py
import logging
import sys
from datetime import datetime
from pathlib import Path

def setup_logging(
    log_level: str = "INFO",
    log_file: str = None,
    enable_console: bool = True
):
    """Configure logging for the application."""

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    logger.handlers.clear()

    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger

# Usage
logger = setup_logging(
    log_level="INFO",
    log_file="logs/agent.log",
    enable_console=True
)
```

---

### Structured Logging

```python
# utils/structured_logger.py
import logging
import json
from datetime import datetime
from typing import Dict, Any

class StructuredLogger:
    """Logger that outputs JSON-structured logs."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log(self, level: str, message: str, **kwargs):
        """Log with structured data."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": level,
            "message": message,
            **kwargs
        }

        self.logger.log(
            getattr(logging, level.upper()),
            json.dumps(log_data)
        )

    def info(self, message: str, **kwargs):
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log("ERROR", message, **kwargs)

    def agent_invocation(
        self,
        session_id: str,
        user_input: str,
        output: str,
        duration: float,
        tokens_used: int,
        tools_used: list
    ):
        """Log agent invocation with all details."""
        self.info(
            "agent_invocation",
            session_id=session_id,
            user_input=user_input[:100],  # Truncate long inputs
            output=output[:200],
            duration_seconds=duration,
            tokens_used=tokens_used,
            tools_used=tools_used,
            event_type="agent_invocation"
        )

    def tool_execution(
        self,
        tool_name: str,
        input_params: Dict,
        output: Any,
        duration: float,
        success: bool,
        error: str = None
    ):
        """Log tool execution."""
        self._log(
            "ERROR" if not success else "INFO",
            "tool_execution",
            tool_name=tool_name,
            input_params=input_params,
            output=str(output)[:200] if success else None,
            duration_seconds=duration,
            success=success,
            error=error,
            event_type="tool_execution"
        )

# Usage
logger = StructuredLogger("agent_app")

logger.agent_invocation(
    session_id="user_123",
    user_input="What's the weather?",
    output="The weather is sunny, 22°C",
    duration=2.5,
    tokens_used=150,
    tools_used=["weather_api"]
)
```

---

### Agent Execution Logging

```python
# agents/logged_agent.py
import time
from typing import Dict, Any
from langchain.agents import AgentExecutor
from langchain.callbacks import get_openai_callback
from utils.structured_logger import StructuredLogger

logger = StructuredLogger("agent")

class LoggedAgentExecutor:
    """Agent executor with comprehensive logging."""

    def __init__(self, agent_executor: AgentExecutor, session_id: str = None):
        self.agent_executor = agent_executor
        self.session_id = session_id or "unknown"

    def invoke(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Execute agent with logging."""

        start_time = time.time()
        user_input = inputs.get("input", "")

        logger.info(
            "agent_started",
            session_id=self.session_id,
            input=user_input[:100]
        )

        try:
            with get_openai_callback() as cb:
                result = self.agent_executor.invoke(inputs)

                duration = time.time() - start_time

                # Extract tools used
                tools_used = []
                for step in result.get("intermediate_steps", []):
                    if len(step) > 0 and hasattr(step[0], 'tool'):
                        tools_used.append(step[0].tool)

                # Log successful execution
                logger.agent_invocation(
                    session_id=self.session_id,
                    user_input=user_input,
                    output=result.get("output", ""),
                    duration=duration,
                    tokens_used=cb.total_tokens,
                    tools_used=tools_used
                )

                # Add metadata to result
                result["metadata"] = {
                    "duration": duration,
                    "tokens_used": cb.total_tokens,
                    "total_cost": cb.total_cost,
                    "tools_used": tools_used
                }

                return result

        except Exception as e:
            duration = time.time() - start_time

            logger.error(
                "agent_failed",
                session_id=self.session_id,
                input=user_input[:100],
                error=str(e),
                duration=duration,
                error_type=type(e).__name__
            )

            raise

# Usage
logged_agent = LoggedAgentExecutor(
    agent_executor=agent_executor,
    session_id="user_123"
)

result = logged_agent.invoke({"input": "Hello"})
print(f"Tokens used: {result['metadata']['tokens_used']}")
```

---

## LangSmith Integration

### Setup LangSmith

```python
# config/settings.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # ... other settings

    # LangSmith configuration
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: str
    langchain_project: str = "my-agent-project"

    class Config:
        env_file = ".env"

# .env file
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
# LANGCHAIN_API_KEY=your-api-key
# LANGCHAIN_PROJECT=my-agent-project
```

---

### Using LangSmith in Code

```python
import os
from langchain.smith import RunEvalConfig
from langsmith import Client

# Set environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "production-agent"

# Agent will automatically send traces to LangSmith
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)

# Add custom metadata to traces
result = agent_executor.invoke(
    {"input": "What's the weather?"},
    config={
        "metadata": {
            "user_id": "user_123",
            "session_id": "session_456",
            "environment": "production"
        }
    }
)

# LangSmith client for custom operations
client = Client()

# List recent runs
runs = client.list_runs(project_name="production-agent", limit=10)

for run in runs:
    print(f"Run ID: {run.id}")
    print(f"Status: {run.status}")
    print(f"Duration: {run.total_tokens}")
```

---

### Custom Callbacks for LangSmith

```python
from langchain.callbacks.base import BaseCallbackHandler
from typing import Any, Dict

class CustomLangSmithCallback(BaseCallbackHandler):
    """Custom callback for additional LangSmith tracking."""

    def on_agent_action(self, action, **kwargs):
        """Called when agent takes an action."""
        print(f"Agent Action: {action.tool} with {action.tool_input}")

    def on_tool_start(self, tool, input_str: str, **kwargs):
        """Called when tool execution starts."""
        print(f"Tool Started: {tool} with input: {input_str}")

    def on_tool_end(self, output: str, **kwargs):
        """Called when tool execution ends."""
        print(f"Tool Output: {output[:100]}")

    def on_tool_error(self, error: Exception, **kwargs):
        """Called when tool encounters an error."""
        print(f"Tool Error: {error}")

# Usage
callbacks = [CustomLangSmithCallback()]

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    callbacks=callbacks
)
```

---

## Custom Tracing

### Request Tracing System

```python
# utils/tracing.py
import uuid
import time
from contextvars import ContextVar
from typing import Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime

# Context variable for trace ID
trace_context: ContextVar[str] = ContextVar("trace_id", default=None)

@dataclass
class TraceSpan:
    """Represents a span in a trace."""
    span_id: str
    name: str
    start_time: float
    end_time: float = None
    duration: float = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    children: List['TraceSpan'] = field(default_factory=list)

class Tracer:
    """Distributed tracing for agent operations."""

    def __init__(self):
        self.traces: Dict[str, List[TraceSpan]] = {}

    def start_trace(self, trace_id: str = None) -> str:
        """Start a new trace."""
        if trace_id is None:
            trace_id = str(uuid.uuid4())

        trace_context.set(trace_id)
        self.traces[trace_id] = []

        return trace_id

    def start_span(self, name: str, **attributes) -> TraceSpan:
        """Start a new span."""
        trace_id = trace_context.get()

        span = TraceSpan(
            span_id=str(uuid.uuid4()),
            name=name,
            start_time=time.time(),
            attributes=attributes
        )

        if trace_id:
            self.traces[trace_id].append(span)

        return span

    def end_span(self, span: TraceSpan):
        """End a span."""
        span.end_time = time.time()
        span.duration = span.end_time - span.start_time

    def get_trace(self, trace_id: str) -> List[TraceSpan]:
        """Get all spans for a trace."""
        return self.traces.get(trace_id, [])

    def export_trace(self, trace_id: str) -> Dict:
        """Export trace as JSON."""
        spans = self.get_trace(trace_id)

        return {
            "trace_id": trace_id,
            "spans": [
                {
                    "span_id": s.span_id,
                    "name": s.name,
                    "duration": s.duration,
                    "attributes": s.attributes
                }
                for s in spans
            ],
            "total_duration": sum(s.duration for s in spans if s.duration)
        }

# Global tracer instance
tracer = Tracer()

# Usage
trace_id = tracer.start_trace()

# Agent invocation span
agent_span = tracer.start_span("agent_invocation", user_id="user_123")

# Tool execution span
tool_span = tracer.start_span("tool_execution", tool_name="calculator")
# ... execute tool
tracer.end_span(tool_span)

# LLM call span
llm_span = tracer.start_span("llm_call", model="gpt-4")
# ... call LLM
tracer.end_span(llm_span)

tracer.end_span(agent_span)

# Export trace
trace_data = tracer.export_trace(trace_id)
print(json.dumps(trace_data, indent=2))
```

---

## Metrics & Analytics

### Metrics Collector

```python
# utils/metrics.py
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
import statistics

@dataclass
class Metric:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime
    tags: Dict[str, str] = None

class MetricsCollector:
    """Collect and aggregate metrics."""

    def __init__(self):
        self.metrics: List[Metric] = []
        self.counters: Dict[str, int] = defaultdict(int)
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = defaultdict(list)

    def increment(self, name: str, value: int = 1, tags: Dict = None):
        """Increment a counter."""
        self.counters[name] += value
        self.metrics.append(Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags
        ))

    def gauge(self, name: str, value: float, tags: Dict = None):
        """Set a gauge value."""
        self.gauges[name] = value
        self.metrics.append(Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags
        ))

    def histogram(self, name: str, value: float, tags: Dict = None):
        """Record a histogram value."""
        self.histograms[name].append(value)
        self.metrics.append(Metric(
            name=name,
            value=value,
            timestamp=datetime.utcnow(),
            tags=tags
        ))

    def get_stats(self, name: str, window: timedelta = None) -> Dict:
        """Get statistics for a metric."""
        values = self.histograms.get(name, [])

        if not values:
            return {}

        # Filter by time window if specified
        if window:
            cutoff = datetime.utcnow() - window
            values = [
                m.value for m in self.metrics
                if m.name == name and m.timestamp > cutoff
            ]

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "min": min(values),
            "max": max(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0
        }

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []

        # Counters
        for name, value in self.counters.items():
            lines.append(f"{name}_total {value}")

        # Gauges
        for name, value in self.gauges.items():
            lines.append(f"{name} {value}")

        # Histograms
        for name, values in self.histograms.items():
            if values:
                stats = self.get_stats(name)
                lines.append(f"{name}_count {stats['count']}")
                lines.append(f"{name}_sum {sum(values)}")
                lines.append(f"{name}_mean {stats['mean']}")

        return "\n".join(lines)

# Global metrics collector
metrics = MetricsCollector()

# Usage
metrics.increment("agent_invocations", tags={"status": "success"})
metrics.histogram("agent_duration", 2.5, tags={"model": "gpt-4"})
metrics.gauge("active_sessions", 42)

# Get statistics
stats = metrics.get_stats("agent_duration")
print(f"Average duration: {stats['mean']:.2f}s")
```

---

### Agent Metrics Integration

```python
# agents/monitored_agent.py
from utils.metrics import metrics

class MonitoredAgentExecutor:
    """Agent executor with metrics collection."""

    def __init__(self, agent_executor: AgentExecutor):
        self.agent_executor = agent_executor

    def invoke(self, inputs: Dict) -> Dict:
        """Execute with metrics."""

        start_time = time.time()

        try:
            # Increment invocation counter
            metrics.increment("agent_invocations_total")

            # Execute agent
            with get_openai_callback() as cb:
                result = self.agent_executor.invoke(inputs)

            # Record metrics
            duration = time.time() - start_time
            metrics.histogram("agent_duration_seconds", duration)
            metrics.histogram("agent_tokens_used", cb.total_tokens)
            metrics.histogram("agent_cost_usd", cb.total_cost)
            metrics.increment("agent_success_total")

            # Track tools used
            tools_used = len(result.get("intermediate_steps", []))
            metrics.histogram("agent_tools_used", tools_used)

            return result

        except Exception as e:
            metrics.increment("agent_errors_total", tags={
                "error_type": type(e).__name__
            })
            raise

# Usage
monitored_agent = MonitoredAgentExecutor(agent_executor)
result = monitored_agent.invoke({"input": "Hello"})

# View metrics
print(metrics.export_prometheus())
```

---

## Alerting

### Alert System

```python
# utils/alerting.py
from typing import Callable, Dict, Any
from dataclasses import dataclass
from enum import Enum
import smtplib
from email.mime.text import MIMEText

class AlertSeverity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Alert:
    """An alert to be sent."""
    severity: AlertSeverity
    title: str
    message: str
    metadata: Dict[str, Any] = None

class AlertManager:
    """Manage and send alerts."""

    def __init__(self):
        self.handlers: Dict[AlertSeverity, List[Callable]] = {
            severity: [] for severity in AlertSeverity
        }
        self.alert_history: List[Alert] = []

    def register_handler(
        self,
        severity: AlertSeverity,
        handler: Callable[[Alert], None]
    ):
        """Register an alert handler."""
        self.handlers[severity].append(handler)

    def send_alert(self, alert: Alert):
        """Send an alert through registered handlers."""
        self.alert_history.append(alert)

        # Call handlers for this severity and higher
        for severity in AlertSeverity:
            if severity.value >= alert.severity.value:
                for handler in self.handlers.get(severity, []):
                    try:
                        handler(alert)
                    except Exception as e:
                        print(f"Alert handler failed: {e}")

    def alert_if(
        self,
        condition: bool,
        severity: AlertSeverity,
        title: str,
        message: str,
        **metadata
    ):
        """Send alert if condition is true."""
        if condition:
            self.send_alert(Alert(
                severity=severity,
                title=title,
                message=message,
                metadata=metadata
            ))

# Alert handlers
def console_alert_handler(alert: Alert):
    """Print alerts to console."""
    print(f"[{alert.severity.value.upper()}] {alert.title}")
    print(f"  {alert.message}")

def email_alert_handler(alert: Alert):
    """Send alert via email."""
    if alert.severity in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
        msg = MIMEText(f"{alert.title}\n\n{alert.message}")
        msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
        msg['From'] = "alerts@yourapp.com"
        msg['To'] = "team@yourapp.com"

        # Send email (configure SMTP server)
        # server = smtplib.SMTP('smtp.gmail.com', 587)
        # server.send_message(msg)

# Global alert manager
alert_manager = AlertManager()
alert_manager.register_handler(AlertSeverity.ERROR, console_alert_handler)
alert_manager.register_handler(AlertSeverity.CRITICAL, email_alert_handler)

# Usage
# Alert on high error rate
error_rate = metrics.counters["agent_errors_total"] / metrics.counters["agent_invocations_total"]
alert_manager.alert_if(
    condition=error_rate > 0.1,
    severity=AlertSeverity.ERROR,
    title="High Agent Error Rate",
    message=f"Error rate is {error_rate:.1%}",
    error_rate=error_rate
)

# Alert on high costs
total_cost = sum(m.value for m in metrics.metrics if m.name == "agent_cost_usd")
alert_manager.alert_if(
    condition=total_cost > 100,
    severity=AlertSeverity.WARNING,
    title="High API Costs",
    message=f"Total cost today: ${total_cost:.2f}",
    total_cost=total_cost
)
```

---

## Production Dashboards

### Metrics Endpoint for Grafana/Prometheus

```python
# main.py or api endpoint
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse

app = FastAPI()

@app.get("/metrics", response_class=PlainTextResponse)
async def prometheus_metrics():
    """Expose metrics in Prometheus format."""
    return metrics.export_prometheus()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "metrics": {
            "total_invocations": metrics.counters["agent_invocations_total"],
            "error_rate": (
                metrics.counters["agent_errors_total"] /
                max(metrics.counters["agent_invocations_total"], 1)
            ),
            "avg_duration": metrics.get_stats("agent_duration_seconds").get("mean", 0)
        }
    }
```

---

## Best Practices

### 1. Log Everything Important

```python
logger.info("agent_started", session_id=session_id)
logger.info("tool_executed", tool_name=tool_name)
logger.error("agent_failed", error=str(e))
```

### 2. Use Structured Logging

```python
# JSON format for easy parsing
{"timestamp": "2026-01-25T10:00:00Z", "level": "INFO", "event": "agent_invocation"}
```

### 3. Track Key Metrics

```python
metrics.histogram("agent_duration_seconds", duration)
metrics.histogram("agent_tokens_used", tokens)
metrics.increment("agent_errors_total")
```

### 4. Set Up Alerts

```python
alert_manager.alert_if(error_rate > 0.1, AlertSeverity.ERROR, ...)
```

### 5. Use Distributed Tracing

```python
trace_id = tracer.start_trace()
# ... track all operations
tracer.export_trace(trace_id)
```

---

_Last Updated: January 25, 2026_
