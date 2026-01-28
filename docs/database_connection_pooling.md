# Database Connection Pooling

## Overview

The application now uses **connection pooling** for optimal database performance and resource management.

## Current Implementation

### Before (Issues)

- ❌ Single connection created at startup
- ❌ No connection pooling
- ❌ No reconnection logic
- ❌ Connection held indefinitely
- ❌ Potential bottleneck under load

### After (Optimized) ✅

- ✅ **Connection Pool** with configurable size
- ✅ **Automatic connection recycling** (every hour by default)
- ✅ **Pre-ping verification** before using connections
- ✅ **Proper lifecycle management** (startup/shutdown)
- ✅ **Health checks** and reconnection capability
- ✅ **Lazy initialization** option

## Connection Pool Configuration

```python
DatabaseManager(
    db_uri="your_connection_string",
    pool_size=5,           # Base connections in pool
    max_overflow=10,       # Additional connections allowed
    pool_recycle=3600,     # Recycle after 1 hour
    lazy_init=False        # Connect immediately
)
```

### Parameters Explained

| Parameter       | Default | Description                                      |
| --------------- | ------- | ------------------------------------------------ |
| `pool_size`     | 5       | Number of persistent connections maintained      |
| `max_overflow`  | 10      | Extra connections created under load (total: 15) |
| `pool_recycle`  | 3600    | Recycle connections after N seconds              |
| `pool_pre_ping` | True    | Test connection health before use                |
| `lazy_init`     | False   | Delay connection until first use                 |

## How It Works

### Connection Lifecycle

```
API Startup
    ↓
Initialize Connection Pool (5 connections)
    ↓
Handle Requests (reuse connections from pool)
    ↓
Under Load: Create additional connections (up to 10)
    ↓
After 1 hour: Recycle old connections
    ↓
API Shutdown: Close all connections
```

### Per-Request Flow

```python
# Request comes in
→ Get connection from pool (if available)
→ Execute SQL query
→ Return connection to pool
→ Response sent

# If pool is full (15 connections in use)
→ Wait for available connection
→ Timeout if wait exceeds threshold
```

## API Endpoints

### Health Checks

```bash
# General health
GET /health

# Database-specific health
GET /health/db
```

Response:

```json
{
  "database_healthy": true,
  "status": "connected"
}
```

### Configuration

```bash
GET /config
```

Shows current pool configuration.

## Benefits

### Performance

- **5-15 concurrent requests** handled efficiently
- **No connection overhead** per request
- **Sub-millisecond** connection acquisition

### Reliability

- **Auto-reconnection** on connection loss
- **Pre-ping verification** prevents stale connections
- **Graceful degradation** under high load

### Resource Management

- **Connection recycling** prevents memory leaks
- **Proper cleanup** on shutdown
- **Configurable limits** prevent resource exhaustion

## Production Recommendations

### Understanding Your Needs

**Important:** 100 users ≠ 100 concurrent connections!

Connection pool sizing depends on:

1. **Request Rate**: How many requests per second?
2. **Query Duration**: How long does each query take?
3. **Concurrency Pattern**: Are requests evenly distributed or bursty?

**Formula for concurrent connections needed:**

```
Concurrent Connections = Requests per Second × Average Query Time (seconds)
```

**Example for 100 Users:**

- Users send 1 request every 30 seconds = 100/30 = 3.3 req/s
- Average query time = 2 seconds
- **Concurrent connections needed = 3.3 × 2 = ~7 connections**
- ✅ **pool_size=5, max_overflow=10 (total 15) is MORE than enough**

### Sizing Guidelines

#### **For 100 Users (Default: pool_size=5, max_overflow=10)**

**✅ SUFFICIENT when:**

- Normal usage: 1-5 requests per second
- Query time: 1-3 seconds
- Expected concurrent connections: 5-15
- **This covers most scenarios**

**⚠️ MIGHT NEED MORE when:**

- Peak traffic bursts (all users hit simultaneously)
- Slow queries (5-10+ seconds each)
- High request rate (10+ req/s sustained)
- Long-running analytics queries

**Recommended for 100 users:**

```python
# Conservative (recommended)
pool_size=10
max_overflow=15
pool_recycle=1800

# Moderate (current default - usually sufficient)
pool_size=5
max_overflow=10
pool_recycle=3600

# Minimal (only if low usage)
pool_size=3
max_overflow=7
pool_recycle=3600
```

### Traffic-Based Recommendations

### For Low Traffic (< 100 req/min, < 2 req/s)

```python
pool_size=3
max_overflow=5
pool_recycle=3600
# Total: 8 connections
# Good for: 10-50 occasional users
```

### For Medium Traffic (100-1000 req/min, 2-15 req/s)

```python
pool_size=10
max_overflow=20
pool_recycle=1800
# Total: 30 connections
# Good for: 100-500 active users
```

### For High Traffic (> 1000 req/min, > 15 req/s)

```python
pool_size=20
max_overflow=40
pool_recycle=900
# Total: 60 connections
# Good for: 500+ active users
```

## How to Test Your Configuration

### 1. Load Testing

**Install load testing tool:**

```bash
pip install locust
```

**Create `locustfile.py`:**

```python
from locust import HttpUser, task, between

class APIUser(HttpUser):
    wait_time = between(1, 3)

    @task
    def query_agent(self):
        self.client.post("/query", json={
            "question": "What is 5 + 7?"
        })

# Run: locust -f locustfile.py --host=http://localhost:8000
# Open http://localhost:8089 and simulate 100 users
```

### 2. Monitor Connection Usage

**Check active connections:**

```python
# Add to your API for monitoring
@app.get("/pool/stats")
async def pool_stats():
    if not agent_app:
        raise HTTPException(status_code=503, detail="Agent not loaded")

    pool = agent_app.db_manager.get_database()._engine.pool
    return {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total": pool.size() + pool.overflow()
    }
```

### 3. Watch for Warning Signs

**Signs you need more connections:**

- ❌ Frequent timeout errors
- ❌ Slow response times during peak
- ❌ Logs show "QueuePool limit exceeded"
- ❌ High checkout wait times

**Signs you have too many:**

- ⚠️ Database server connection limit reached
- ⚠️ Most connections idle
- ⚠️ Memory usage unnecessarily high

## Real-World Scenarios

### Scenario 1: 100 Users, Chat Application

- Users send message every 30-60 seconds
- Request rate: ~2 req/s
- Query time: 1-2 seconds
- **Needed: ~4 concurrent connections**
- ✅ **pool_size=5, max_overflow=10 = PERFECT**

### Scenario 2: 100 Users, Dashboard (Frequent Polling)

- Dashboard refreshes every 5 seconds
- Request rate: 20 req/s
- Query time: 0.5 seconds
- **Needed: ~10 concurrent connections**
- ⚠️ **pool_size=5, max_overflow=10 = ADEQUATE** (at capacity)
- 💡 **Recommended: pool_size=10, max_overflow=15**

### Scenario 3: 100 Users, Analytics (Long Queries)

- Users run reports occasionally
- Request rate: 1 req/s
- Query time: 10-30 seconds
- **Needed: 10-30 concurrent connections**
- ❌ **pool_size=5, max_overflow=10 = INSUFFICIENT**
- 💡 **Recommended: pool_size=20, max_overflow=30**

### Scenario 4: 100 Users, Peak Burst Traffic

- All 100 users hit API simultaneously
- Request rate: Burst of 100 req in 1 second
- Query time: 2 seconds
- **Needed: Potentially 100+ connections temporarily**
- ❌ **pool_size=5, max_overflow=10 = INSUFFICIENT**
- 💡 **Solutions:**
  - Increase pool: pool_size=30, max_overflow=70
  - Add rate limiting
  - Use request queuing
  - Scale horizontally (multiple API instances)

## Monitoring

### Log Messages

```
[INFO] Database connection initialized with pool_size=5
[INFO] Database connection pool disposed
[ERROR] Database health check failed: <error>
```

### Metrics to Monitor

- Pool checkout time
- Number of active connections
- Connection timeouts
- Failed health checks

## Troubleshooting

### Issue: "Too many connections"

**Solution:** Reduce `pool_size` or `max_overflow`

### Issue: "Connection timeout"

**Solution:** Increase `max_overflow` or optimize queries

### Issue: "Stale connections"

**Solution:** Decrease `pool_recycle` value

### Issue: "Slow first request"

**Solution:** Set `lazy_init=False` (default)

## Code Examples

### Manual Reconnection

```python
# Force reconnect to database
agent_app.db_manager.reconnect()
```

### Context Manager Usage

```python
with agent_app.db_manager.get_connection() as db:
    result = db.run("SELECT * FROM users")
```

### Health Check

```python
is_healthy = agent_app.db_manager.health_check()
if not is_healthy:
    agent_app.db_manager.reconnect()
```

## Summary

✅ **No new connections per request** - connections are pooled and reused  
✅ **Optimal for production** - handles concurrent requests efficiently  
✅ **Auto-recovery** - reconnects on failures  
✅ **Configurable** - tune for your workload  
✅ **Observable** - health checks and logging
