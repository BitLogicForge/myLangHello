# Connection Pool Size Calculator

## Quick Formula

```
Required Pool Size = (Requests per Second) × (Average Query Time in Seconds)
```

Add 20-50% buffer for safety:

```
pool_size = Required Pool Size × 0.4
max_overflow = Required Pool Size × 0.6
Total = pool_size + max_overflow
```

## Examples for 100 Users

### Example 1: Low Activity

**Pattern:** Users make 1 request every minute

- Total requests per second: 100 / 60 = **1.67 req/s**
- Average query time: **2 seconds**
- Required: 1.67 × 2 = **3.34 connections**
- With 50% buffer: **5 connections**

**Recommended:**

```python
pool_size=3
max_overflow=5
# Total: 8 connections ✅
```

### Example 2: Moderate Activity (Most Common)

**Pattern:** Users make 1 request every 30 seconds

- Total requests per second: 100 / 30 = **3.33 req/s**
- Average query time: **2 seconds**
- Required: 3.33 × 2 = **6.66 connections**
- With 50% buffer: **10 connections**

**Recommended:**

```python
pool_size=5
max_overflow=10
# Total: 15 connections ✅
```

### Example 3: High Activity

**Pattern:** Users make 1 request every 10 seconds

- Total requests per second: 100 / 10 = **10 req/s**
- Average query time: **2 seconds**
- Required: 10 × 2 = **20 connections**
- With 50% buffer: **30 connections**

**Recommended:**

```python
pool_size=15
max_overflow=20
# Total: 35 connections ✅
```

### Example 4: Very High Activity / Dashboard

**Pattern:** Dashboard refreshes every 5 seconds

- Total requests per second: 100 / 5 = **20 req/s**
- Average query time: **1 second**
- Required: 20 × 1 = **20 connections**
- With 50% buffer: **30 connections**

**Recommended:**

```python
pool_size=15
max_overflow=20
# Total: 35 connections ✅
```

### Example 5: Slow Queries

**Pattern:** Users make 1 request every 30 seconds (but queries are slow)

- Total requests per second: 100 / 30 = **3.33 req/s**
- Average query time: **10 seconds** (slow analytics)
- Required: 3.33 × 10 = **33.3 connections**
- With 50% buffer: **50 connections**

**Recommended:**

```python
pool_size=25
max_overflow=30
# Total: 55 connections ✅
```

## Decision Matrix

| Users | Req/User/Min | Total Req/s | Query Time | Required | pool_size | max_overflow |
| ----- | ------------ | ----------- | ---------- | -------- | --------- | ------------ |
| 100   | 1            | 1.67        | 2s         | 3        | 3         | 5            |
| 100   | 2            | 3.33        | 2s         | 7        | 5         | 10           |
| 100   | 6            | 10          | 2s         | 20       | 10        | 15           |
| 100   | 12           | 20          | 1s         | 20       | 15        | 20           |
| 100   | 2            | 3.33        | 10s        | 33       | 20        | 25           |
| 500   | 2            | 16.67       | 2s         | 33       | 20        | 25           |
| 1000  | 2            | 33.33       | 2s         | 67       | 35        | 40           |

## Your Current Configuration

**Default in code:**

```python
pool_size=10
max_overflow=15
pool_recycle=1800
# Total: 25 connections
```

**This handles:**

- Up to **12.5 requests per second** (with 2-second queries)
- Or **~100 users** making requests every 15-20 seconds
- Or **~100 users** with bursts up to 25 concurrent requests

## Is Your Config Enough?

### ✅ YES - Current config (pool=10, overflow=15) is ENOUGH if:

- Users make 1-4 requests per minute (normal web app usage)
- Queries complete in 1-3 seconds
- Traffic is relatively evenly distributed
- You have **~100 typical users**

### ⚠️ MAYBE - Monitor closely if:

- Users make 5-10 requests per minute (active usage)
- Occasional slow queries (3-5 seconds)
- Some peak periods with traffic bursts
- **Watch logs for timeouts or "pool exhausted" errors**

### ❌ NO - You need more connections if:

- Users make 10+ requests per minute (dashboard/polling)
- Queries take 5-10+ seconds (analytics)
- All users hit API simultaneously (peak bursts)
- Logs show connection timeouts or pool exhaustion
- **Increase to pool_size=20, max_overflow=30**

## How to Measure Your Actual Usage

### Step 1: Add Monitoring Endpoint

```python
# In api.py
@app.get("/metrics/pool")
async def pool_metrics():
    pool = agent_app.db_manager.get_database()._engine.pool
    return {
        "size": pool.size(),
        "checked_out": pool.checkedout(),
        "checked_in": pool.checkedin(),
        "overflow": pool.overflow(),
        "utilization": f"{pool.checkedout() / (pool.size() + pool.overflow()) * 100:.1f}%"
    }
```

### Step 2: Monitor During Peak Usage

**Good signs (✅):**

- Utilization < 80%
- checked_out < total capacity
- No timeout errors in logs

**Warning signs (⚠️):**

- Utilization > 80% sustained
- checked_out reaches max frequently
- Occasional timeouts

**Critical signs (❌):**

- Utilization at 100%
- Timeout errors in logs
- Users reporting slow responses

## Quick Adjustment Guide

### If you see timeouts:

```python
# Double your pool
pool_size=20       # was 10
max_overflow=30    # was 15
```

### If you see "QueuePool limit exceeded":

```python
# Increase overflow
pool_size=10
max_overflow=30    # was 15
```

### If queries are slow (5+ seconds):

```python
# Increase both significantly
pool_size=25
max_overflow=35
```

### If most connections idle:

```python
# Reduce to save resources
pool_size=5
max_overflow=8
```

## Summary for 100 Users

**Your current configuration (pool_size=10, max_overflow=15, total=25):**

✅ **SUFFICIENT** for:

- Normal web application usage
- 3-5 requests per user per minute
- 1-3 second query times
- This is a **good conservative default for 100 users**

**Monitor and increase if:**

- Users are very active (10+ req/min per user)
- You have analytics/slow queries
- You see timeouts or high pool utilization
- Peak burst traffic patterns

**Bottom line:** Your current config is **well-sized for 100 typical users**! 🎯
