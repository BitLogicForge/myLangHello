# Logging Implementation Guide

## Overview

Comprehensive logging has been implemented across all services, API, and agent components for better observability, debugging, and monitoring.

## Logging Architecture

### Components with Logging

✅ **API Layer** ([api.py](api.py))

- Request/response tracking
- Error logging with stack traces
- Startup/shutdown events
- Health check monitoring

✅ **Agent Application** ([base_chain_agen_func_tooling.py](base_chain_agen_func_tooling.py))

- Component initialization
- Agent execution flow
- Configuration details

✅ **Services Layer**

- [database_manager.py](services/database_manager.py) - Connection pooling, health checks
- [agent_factory.py](services/agent_factory.py) - Agent/executor creation
- [prompt_builder.py](services/prompt_builder.py) - Prompt loading and building
- [tools_manager.py](services/tools_manager.py) - Tool registration and configuration

## Log Levels Used

| Level       | Usage                          | Examples                                                  |
| ----------- | ------------------------------ | --------------------------------------------------------- |
| **DEBUG**   | Detailed diagnostic info       | "Loading system prompt...", "Creating agent..."           |
| **INFO**    | General informational events   | "Agent initialized successfully", "Query completed"       |
| **WARNING** | Warning messages               | "Database health check failed", "LangServe not installed" |
| **ERROR**   | Error events with stack traces | "Agent execution failed", "Database connection error"     |

## Configuration

### API Logging Configuration

```python
# In api.py
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),      # Console output
        logging.FileHandler('api.log') # File output
    ]
)
```

### Output Format

```
2026-01-28 10:30:45,123 - __main__ - INFO - Initializing agent application...
2026-01-28 10:30:45,456 - services.database_manager - INFO - Database connection initialized with pool_size=5
2026-01-28 10:30:45,789 - services.tools_manager - INFO - ToolsManager initialized with 12 tools
```

## Log Files

### api.log

- All API-level events
- Request/response logs
- Startup/shutdown events
- Health checks

**Location:** `e:\workoff\Ai\myLangHello\api.log`

## Key Logging Events

### Application Startup

```
INFO - Initializing agent application...
INFO - Setting up database manager...
INFO - Database connection initialized with pool_size=5
INFO - Setting up tools manager...
INFO - ToolsManager initialized with 12 tools
INFO - Building prompt template...
INFO - Prompt template built successfully
INFO - Creating agent executor...
INFO - Agent executor created successfully
INFO - ✅ AgentApp initialized successfully
INFO - Application startup initiated
INFO - ✅ Database connection pool initialized and healthy
INFO - Application startup complete
```

### Query Processing

```
INFO - Processing query (session: abc123)
DEBUG - Question: How many users are registered...
DEBUG - Included 3 history messages
INFO - Query completed successfully (session: abc123)
```

### Error Handling

```
ERROR - Agent execution failed: Connection timeout
Traceback (most recent call last):
  File "base_chain_agen_func_tooling.py", line 156, in run
    response = self.agent_executor.invoke({"input": question})
  ...
```

### Database Operations

```
INFO - Database connection initialized with pool_size=5
DEBUG - SQL toolkit created with 4 tools
INFO - Database connection pool disposed
ERROR - Database health check failed: Unable to connect
```

### Application Shutdown

```
INFO - Application shutdown initiated
INFO - ✅ Database connection pool closed
INFO - Application shutdown complete
```

## Logging Best Practices

### 1. Component Initialization

```python
logger.info(f"ComponentName initialized with {param_count} items")
```

### 2. Method Entry (Debug Level)

```python
logger.debug("Starting important_operation...")
```

### 3. Success Confirmation

```python
logger.info("✅ Operation completed successfully")
```

### 4. Warning Conditions

```python
logger.warning("⚠️ Non-critical issue detected")
```

### 5. Error Handling

```python
try:
    # operation
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
    raise
```

## Monitoring and Debugging

### Useful Log Queries

**Check initialization sequence:**

```bash
grep "initialized" api.log
```

**Find errors:**

```bash
grep "ERROR" api.log
```

**Track specific session:**

```bash
grep "session: abc123" api.log
```

**Monitor database health:**

```bash
grep "Database" api.log | grep -E "health|connection"
```

### Performance Tracking

Logs include timing information for:

- Application startup
- Component initialization
- Query processing
- Database operations

Example:

```
2026-01-28 10:30:45,123 - INFO - Processing query (session: abc123)
2026-01-28 10:30:47,456 - INFO - Query completed successfully (session: abc123)
# Duration: ~2.3 seconds
```

## Customization

### Change Log Level

**For more verbose logging (development):**

```python
logging.basicConfig(level=logging.DEBUG)
```

**For production (less verbose):**

```python
logging.basicConfig(level=logging.WARNING)
```

### Add Custom Handler

**Rotate log files:**

```python
from logging.handlers import RotatingFileHandler

handler = RotatingFileHandler(
    'api.log',
    maxBytes=10_000_000,  # 10MB
    backupCount=5
)
logging.basicConfig(handlers=[handler])
```

**JSON logging:**

```python
import json
import logging

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        })

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
```

### Environment-Based Configuration

```python
import os

log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(level=getattr(logging, log_level))
```

## Production Recommendations

### 1. Structured Logging

- Use JSON format for easy parsing
- Include request IDs for tracing
- Add contextual information

### 2. Log Aggregation

- Use ELK Stack (Elasticsearch, Logstash, Kibana)
- Or Grafana Loki
- Or cloud-native solutions (CloudWatch, Azure Monitor)

### 3. Log Retention

- Rotate logs daily or by size
- Keep 30-90 days of logs
- Archive older logs to cold storage

### 4. Alerts

Set up alerts for:

- ERROR level logs
- Database connection failures
- High error rates (> 5% of requests)
- Agent execution timeouts

### 5. Privacy

- Sanitize sensitive data in logs
- Don't log passwords, API keys, or PII
- Use log filters for sensitive fields

## Debugging Tips

### Common Issues

**Issue: Too many logs**

```python
# Reduce log level for specific modules
logging.getLogger('langchain').setLevel(logging.WARNING)
logging.getLogger('openai').setLevel(logging.WARNING)
```

**Issue: Missing context**

```python
# Add context to logs
logger.info(f"Query failed", extra={
    'session_id': session_id,
    'user_id': user_id,
    'query_length': len(question)
})
```

**Issue: Performance impact**

```python
# Use conditional logging for expensive operations
if logger.isEnabledFor(logging.DEBUG):
    logger.debug(f"Expensive operation: {compute_expensive_info()}")
```

## Example: Full Request Trace

```
2026-01-28 10:30:45,123 - __main__ - INFO - Processing query (session: abc123)
2026-01-28 10:30:45,124 - __main__ - DEBUG - Question: How many users are registered...
2026-01-28 10:30:45,125 - __main__ - DEBUG - Included 2 history messages
2026-01-28 10:30:45,200 - services.tools_manager - DEBUG - SQL query tool invoked
2026-01-28 10:30:46,100 - services.database_manager - DEBUG - Query executed successfully
2026-01-28 10:30:47,456 - __main__ - INFO - Query completed successfully (session: abc123)
```

## Summary

✅ **Comprehensive logging** across all components  
✅ **Structured format** with timestamps and levels  
✅ **Error tracking** with stack traces  
✅ **Performance monitoring** capability  
✅ **Production-ready** with file output  
✅ **Configurable** log levels and formats

Logging provides full visibility into application behavior, making debugging and monitoring significantly easier.
