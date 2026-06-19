"""Redis-based distributed portfolio cache for production scale."""

import asyncio
import json
from typing import Optional, Dict
from datetime import timedelta

try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not available. Install with: pip install redis")


class RedisPortfolioCache:
    """Production-ready distributed cache using Redis."""

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "portfolio:",
        ttl_minutes: int = 30,
        max_connections: int = 50
    ):
        if not REDIS_AVAILABLE:
            raise ImportError("Redis library not installed")

        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_connections = max_connections
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None

    async def _get_client(self) -> redis.Redis:
        """Get Redis client with connection pooling."""
        if self._client is None:
            self._pool = redis.ConnectionPool.from_url(
                self.redis_url,
                max_connections=self.max_connections,
                decode_responses=True
            )
            self._client = redis.Redis(connection_pool=self._pool)
        return self._client

    def _make_key(self, portfolio_id: str) -> str:
        """Create Redis key with prefix."""
        return f"{self.key_prefix}{portfolio_id}"

    async def get(self, portfolio_id: str) -> Optional[dict]:
        """Get portfolio from Redis cache."""
        try:
            client = await self._get_client()
            key = self._make_key(portfolio_id)
            data = await client.get(key)

            if data:
                # Update access time by refreshing TTL
                await client.expire(key, int(self.ttl.total_seconds()))
                return json.loads(data)

            return None

        except Exception as e:
            print(f"Redis get error: {e}")
            return None

    async def set(self, portfolio_id: str, portfolio_data: dict) -> bool:
        """Store portfolio in Redis with automatic TTL."""
        try:
            client = await self._get_client()
            key = self._make_key(portfolio_id)
            data = json.dumps(portfolio_data)

            await client.setex(
                key,
                int(self.ttl.total_seconds()),
                data
            )
            return True

        except Exception as e:
            print(f"Redis set error: {e}")
            return False

    async def delete(self, portfolio_id: str) -> bool:
        """Delete portfolio from cache."""
        try:
            client = await self._get_client()
            key = self._make_key(portfolio_id)
            result = await client.delete(key)
            return result > 0

        except Exception as e:
            print(f"Redis delete error: {e}")
            return False

    async def get_multiple(self, portfolio_ids: list[str]) -> Dict[str, dict]:
        """Batch get multiple portfolios for efficiency."""
        try:
            client = await self._get_client()
            keys = [self._make_key(pid) for pid in portfolio_ids]
            values = await client.mget(keys)

            results = {}
            for pid, value in zip(portfolio_ids, values):
                if value:
                    results[pid] = json.loads(value)

            return results

        except Exception as e:
            print(f"Redis batch get error: {e}")
            return {}

    async def set_multiple(self, portfolios: Dict[str, dict]) -> bool:
        """Batch set multiple portfolios for efficiency."""
        try:
            client = await self._get_client()
            pipeline = client.pipeline()

            for pid, data in portfolios.items():
                key = self._make_key(pid)
                pipeline.setex(
                    key,
                    int(self.ttl.total_seconds()),
                    json.dumps(data)
                )

            await pipeline.execute()
            return True

        except Exception as e:
            print(f"Redis batch set error: {e}")
            return False

    async def clear_all(self) -> bool:
        """Clear all portfolio caches (use carefully)."""
        try:
            client = await self._get_client()
            pattern = f"{self.key_prefix}*"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await client.delete(*keys)

            return True

        except Exception as e:
            print(f"Redis clear error: {e}")
            return False

    async def get_stats(self) -> dict:
        """Get cache statistics."""
        try:
            client = await self._get_client()
            pattern = f"{self.key_prefix}*"
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)

            return {
                "type": "redis",
                "portfolio_count": len(keys),
                "ttl_minutes": self.ttl.total_seconds() / 60,
                "max_connections": self.max_connections
            }

        except Exception as e:
            return {"error": str(e)}

    async def close(self) -> None:
        """Close Redis connections."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.aclose()


# Global Redis cache instance
redis_cache: Optional[RedisPortfolioCache] = None


async def get_redis_cache() -> RedisPortfolioCache:
    """Get or create global Redis cache instance."""
    global redis_cache
    if redis_cache is None:
        redis_cache = RedisPortfolioCache(
            redis_url="redis://localhost:6379",
            ttl_minutes=30,
            max_connections=100  # Handle 100+ concurrent users
        )
    return redis_cache


async def get_cached_portfolio_redis(portfolio_id: str) -> Optional[dict]:
    """Get portfolio from Redis cache with fallback."""
    cache = await get_redis_cache()
    return await cache.get(portfolio_id)


async def update_cached_portfolio_redis(portfolio_id: str, portfolio_data: dict) -> bool:
    """Update portfolio in Redis cache."""
    cache = await get_redis_cache()
    return await cache.set(portfolio_id, portfolio_data)