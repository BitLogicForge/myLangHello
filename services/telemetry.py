"""Self-hosted telemetry and metrics tracking using Prometheus."""

import logging
import time
from functools import wraps
from typing import Optional, Callable
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Info,
    start_http_server,
)
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class TelemetryManager:
    """Self-hosted metrics collection and monitoring."""

    def __init__(
        self,
        service_name: str = "chatbot-agent",
        metrics_port: int = 9090,
        enable_metrics_server: bool = True,
    ):
        """
        Initialize telemetry manager.

        Args:
            service_name: Name of the service for identification
            metrics_port: Port for Prometheus metrics endpoint
            enable_metrics_server: Whether to start HTTP server for metrics
        """
        self.service_name = service_name
        self.metrics_port = metrics_port

        # Service information
        self.service_info = Info("service", "Service information")
        self.service_info.info(
            {
                "name": service_name,
                "version": "1.0.0",
            }
        )

        # Request metrics
        self.request_total = Counter(
            "agent_requests_total",
            "Total number of agent requests",
            ["endpoint", "status"],
        )

        self.request_duration = Histogram(
            "agent_request_duration_seconds",
            "Agent request duration in seconds",
            ["endpoint"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
        )

        self.requests_in_progress = Gauge(
            "agent_requests_in_progress",
            "Number of requests currently being processed",
            ["endpoint"],
        )

        # LLM metrics
        self.llm_calls_total = Counter(
            "llm_calls_total",
            "Total number of LLM API calls",
            ["model", "provider"],
        )

        self.llm_tokens_total = Counter(
            "llm_tokens_total",
            "Total number of tokens used",
            ["model", "token_type"],  # token_type: input, output, total
        )

        self.llm_duration = Histogram(
            "llm_call_duration_seconds",
            "LLM API call duration in seconds",
            ["model"],
            buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0),
        )

        self.llm_cost_total = Counter(
            "llm_cost_dollars_total",
            "Total estimated cost in dollars",
            ["model"],
        )

        # Agent metrics
        self.agent_iterations = Histogram(
            "agent_iterations_count",
            "Number of reasoning steps per agent execution",
            buckets=(1, 2, 3, 5, 7, 10, 15, 20),
        )

        self.agent_errors = Counter(
            "agent_errors_total",
            "Total number of agent errors",
            ["error_type"],
        )

        # Tool metrics
        self.tool_calls_total = Counter(
            "tool_calls_total",
            "Total number of tool invocations",
            ["tool_name", "status"],
        )

        self.tool_duration = Histogram(
            "tool_call_duration_seconds",
            "Tool execution duration in seconds",
            ["tool_name"],
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
        )

        # Database metrics
        self.db_queries_total = Counter(
            "db_queries_total",
            "Total number of database queries",
            ["status"],
        )

        self.db_query_duration = Histogram(
            "db_query_duration_seconds",
            "Database query duration in seconds",
            buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0),
        )

        self.db_connections_active = Gauge(
            "db_connections_active",
            "Number of active database connections",
        )

        self.db_pool_size = Gauge(
            "db_pool_size",
            "Database connection pool size",
        )

        # Start metrics server
        if enable_metrics_server:
            self._start_metrics_server()

    def _start_metrics_server(self) -> None:
        """Start HTTP server for Prometheus metrics scraping."""
        try:
            start_http_server(self.metrics_port)
            logger.info(f"✅ Metrics server started on port {self.metrics_port}")
            logger.info(
                f"📊 Prometheus metrics available at: http://localhost:{self.metrics_port}/metrics"
            )
        except OSError as e:
            logger.warning(f"Metrics server already running or port in use: {e}")

    @contextmanager
    def track_request(self, endpoint: str):
        """
        Context manager to track request metrics.

        Usage:
            with telemetry.track_request("query"):
                # your code
        """
        self.requests_in_progress.labels(endpoint=endpoint).inc()
        start_time = time.time()
        status = "success"

        try:
            yield
        except Exception as e:
            status = "error"
            self.agent_errors.labels(error_type=type(e).__name__).inc()
            raise
        finally:
            duration = time.time() - start_time
            self.request_duration.labels(endpoint=endpoint).observe(duration)
            self.request_total.labels(endpoint=endpoint, status=status).inc()
            self.requests_in_progress.labels(endpoint=endpoint).dec()

    def track_llm_call(
        self,
        model: str,
        provider: str,
        input_tokens: int,
        output_tokens: int,
        duration: float,
        cost: Optional[float] = None,
    ) -> None:
        """
        Track LLM API call metrics.

        Args:
            model: Model name (e.g., "gpt-4")
            provider: Provider name (e.g., "azure", "openai")
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            duration: Call duration in seconds
            cost: Estimated cost in dollars
        """
        self.llm_calls_total.labels(model=model, provider=provider).inc()
        self.llm_tokens_total.labels(model=model, token_type="input").inc(input_tokens)
        self.llm_tokens_total.labels(model=model, token_type="output").inc(output_tokens)
        self.llm_tokens_total.labels(model=model, token_type="total").inc(
            input_tokens + output_tokens
        )
        self.llm_duration.labels(model=model).observe(duration)

        if cost is not None:
            self.llm_cost_total.labels(model=model).inc(cost)

    def track_tool_call(self, tool_name: str, duration: float, success: bool = True) -> None:
        """
        Track tool invocation metrics.

        Args:
            tool_name: Name of the tool
            duration: Execution duration in seconds
            success: Whether the tool call succeeded
        """
        status = "success" if success else "error"
        self.tool_calls_total.labels(tool_name=tool_name, status=status).inc()
        self.tool_duration.labels(tool_name=tool_name).observe(duration)

    def track_agent_iterations(self, count: int) -> None:
        """Track number of agent reasoning steps."""
        self.agent_iterations.observe(count)

    def track_db_query(self, duration: float, success: bool = True) -> None:
        """Track database query metrics."""
        status = "success" if success else "error"
        self.db_queries_total.labels(status=status).inc()
        self.db_query_duration.observe(duration)

    def update_db_pool_metrics(self, active: int, pool_size: int) -> None:
        """Update database connection pool metrics."""
        self.db_connections_active.set(active)
        self.db_pool_size.set(pool_size)

    def decorator_track_time(self, metric_name: str) -> Callable[[Callable], Callable]:
        """
        Decorator to automatically track function execution time.

        Usage:
            @telemetry.decorator_track_time("my_function")
            def my_function():
                pass
        """

        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    logger.debug(f"{metric_name} executed in {duration:.3f}s")

            return wrapper

        return decorator


# Global telemetry instance (lazy initialization)
_telemetry_instance: Optional[TelemetryManager] = None


def get_telemetry(
    service_name: str = "chatbot-agent",
    metrics_port: int = 9090,
    enable_metrics_server: bool = True,
) -> TelemetryManager:
    """
    Get or create global telemetry instance.

    Args:
        service_name: Name of the service
        metrics_port: Port for metrics endpoint
        enable_metrics_server: Whether to start metrics server

    Returns:
        TelemetryManager instance
    """
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryManager(
            service_name=service_name,
            metrics_port=metrics_port,
            enable_metrics_server=enable_metrics_server,
        )
    return _telemetry_instance
