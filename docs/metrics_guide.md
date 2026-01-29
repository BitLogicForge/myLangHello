# Self-Hosted Metrics and Monitoring Guide

## Overview

This project includes a fully **self-hosted** metrics system using Prometheus. All data stays on your infrastructure - no external services required.

## Components

### 1. Prometheus Metrics Endpoint

**Location:** `http://localhost:9090/metrics`

Automatically started when the API server launches. Prometheus can scrape this endpoint to collect metrics.

### 2. Available Metrics

#### Request Metrics

- `agent_requests_total` - Total requests (labeled by endpoint, status)
- `agent_request_duration_seconds` - Request latency histogram
- `agent_requests_in_progress` - Current active requests

#### LLM Metrics

- `llm_calls_total` - Total LLM API calls (by model, provider)
- `llm_tokens_total` - Token usage (input/output/total)
- `llm_call_duration_seconds` - LLM API latency
- `llm_cost_dollars_total` - Estimated costs

#### Agent Metrics

- `agent_iterations_count` - Reasoning steps per execution
- `agent_errors_total` - Errors by type

#### Tool Metrics

- `tool_calls_total` - Tool invocations (by name, status)
- `tool_call_duration_seconds` - Tool execution time

#### Database Metrics

- `db_queries_total` - Database queries
- `db_query_duration_seconds` - Query latency
- `db_connections_active` - Active connections
- `db_pool_size` - Connection pool size

## Setup Instructions

### Step 1: Install Dependencies

```bash
pip install prometheus-client
```

Already added to `requirements.txt`.

### Step 2: Verify Metrics Endpoint

Start your API:

```bash
python api.py
```

Check metrics:

```bash
curl http://localhost:9090/metrics
```

You should see Prometheus format metrics:

```
# HELP agent_requests_total Total number of agent requests
# TYPE agent_requests_total counter
agent_requests_total{endpoint="query",status="success"} 5.0
```

### Step 3: Install Prometheus (Optional)

**Download Prometheus:**

```bash
# Windows
# Download from https://prometheus.io/download/
# Extract to e.g., C:\prometheus
```

**Configure Prometheus** (`prometheus.yml`):

```yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "chatbot-agent"
    static_configs:
      - targets: ["localhost:9090"]
```

**Run Prometheus:**

```bash
# Windows
cd C:\prometheus
prometheus.exe --config.file=prometheus.yml
```

**Access Prometheus UI:** `http://localhost:9090` (Prometheus default port)

### Step 4: Install Grafana (Optional)

**Download Grafana:**

```bash
# Windows: https://grafana.com/grafana/download
```

**Run Grafana:**

```bash
# Windows
cd C:\grafana\bin
grafana-server.exe
```

**Access Grafana:** `http://localhost:3000`

- Default login: admin/admin

**Add Prometheus Data Source:**

1. Configuration → Data Sources → Add data source
2. Select Prometheus
3. URL: `http://localhost:9090`
4. Save & Test

**Import Dashboard:**
Create panels for:

- Request rate over time
- Average latency (p50, p95, p99)
- Token usage trends
- Cost over time
- Error rates

## Usage Examples

### In Your Code

```python
from services.telemetry import get_telemetry

telemetry = get_telemetry()

# Track a request
with telemetry.track_request("my_endpoint"):
    # Your code here
    pass

# Track LLM call
telemetry.track_llm_call(
    model="gpt-4",
    provider="azure",
    input_tokens=100,
    output_tokens=50,
    duration=1.5,
    cost=0.012
)

# Track tool usage
telemetry.track_tool_call(
    tool_name="sql_query",
    duration=0.3,
    success=True
)
```

### Cost Tracking

```python
from services.cost_tracker import CostTracker

tracker = CostTracker()

# Track a call
cost = tracker.track_call(
    model="gpt-4",
    input_tokens=1000,
    output_tokens=500
)
print(f"Cost: ${cost:.4f}")

# Get statistics
stats = tracker.get_stats()
print(f"Total cost: ${stats['total_cost']}")
print(f"Total calls: {stats['total_calls']}")

# Estimate monthly costs
monthly = CostTracker.estimate_monthly_cost(
    requests_per_day=1000,
    avg_input_tokens=500,
    avg_output_tokens=200,
    model="gpt-4"
)
print(f"Estimated monthly: ${monthly}")
```

## Querying Metrics

### Prometheus Query Examples

```promql
# Request rate (per second)
rate(agent_requests_total[5m])

# Average request duration
rate(agent_request_duration_seconds_sum[5m]) / rate(agent_request_duration_seconds_count[5m])

# 95th percentile latency
histogram_quantile(0.95, agent_request_duration_seconds_bucket)

# Token usage rate
rate(llm_tokens_total[5m])

# Error rate
rate(agent_errors_total[5m])

# Cost per day
increase(llm_cost_dollars_total[1d])
```

## Alerting

### Prometheus Alert Rules

Create `alert.rules.yml`:

```yaml
groups:
  - name: chatbot_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(agent_errors_total[5m]) > 0.1
        annotations:
          summary: "High error rate detected"

      - alert: HighLatency
        expr: histogram_quantile(0.95, agent_request_duration_seconds_bucket) > 10
        annotations:
          summary: "95th percentile latency > 10s"

      - alert: HighCost
        expr: increase(llm_cost_dollars_total[1h]) > 10
        annotations:
          summary: "Cost > $10/hour"
```

## Architecture

```
┌─────────────────┐
│  FastAPI App    │
│  (Port 8000)    │
└────────┬────────┘
         │
         │ exposes
         ▼
┌─────────────────┐
│ Metrics Endpoint│  ◄─── Prometheus scrapes
│ (Port 9090)     │       every 15s
└─────────────────┘
         │
         │ stores in
         ▼
┌─────────────────┐
│   Prometheus    │  ◄─── Grafana queries
│  (Port 9090)    │
└─────────────────┘
         │
         │ visualizes
         ▼
┌─────────────────┐
│    Grafana      │  ◄─── You view dashboards
│  (Port 3000)    │
└─────────────────┘
```

## Benefits

✅ **Fully self-hosted** - No external services
✅ **No data leaves your infrastructure**
✅ **Industry standard** (Prometheus/Grafana)
✅ **Low overhead** - Minimal performance impact
✅ **Production ready** - Battle-tested stack
✅ **Free and open source**

## Next Steps

1. **Basic:** Just run the API - metrics auto-exposed
2. **Intermediate:** Install Prometheus for time-series storage
3. **Advanced:** Add Grafana for beautiful dashboards
4. **Production:** Set up alerts and monitoring automation

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [PromQL Query Language](https://prometheus.io/docs/prometheus/latest/querying/basics/)
