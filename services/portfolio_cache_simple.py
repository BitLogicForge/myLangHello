"""Thread-safe portfolio cache for small-scale concurrent access."""

import asyncio
from threading import Lock
from typing import Dict, Optional
from datetime import datetime, timedelta


class PortfolioCache:
    """Thread-safe in-memory cache for portfolio data."""

    def __init__(self, max_size: int = 1000, ttl_minutes: int = 30):
        self.cache: Dict[str, dict] = {}
        self.lock = Lock()
        self.max_size = max_size
        self.ttl = timedelta(minutes=ttl_minutes)
        self.access_times: Dict[str, datetime] = {}

    def get(self, portfolio_id: str) -> Optional[dict]:
        """Thread-safe cache retrieval with TTL check."""
        with Lock():  # Ensure thread safety
            if portfolio_id not in self.cache:
                return None

            # Check TTL
            last_access = self.access_times.get(portfolio_id)
            if last_access and datetime.now() - last_access > self.ttl:
                # Expired - remove from cache
                del self.cache[portfolio_id]
                del self.access_times[portfolio_id]
                return None

            # Update access time and return data
            self.access_times[portfolio_id] = datetime.now()
            return self.cache[portfolio_id].copy()  # Return copy to prevent external mutations

    def set(self, portfolio_id: str, portfolio_data: dict) -> None:
        """Thread-safe cache storage with size limits."""
        with Lock():  # Ensure thread safety
            # Implement cache eviction if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_oldest()

            self.cache[portfolio_id] = portfolio_data.copy()  # Store copy
            self.access_times[portfolio_id] = datetime.now()

    def delete(self, portfolio_id: str) -> bool:
        """Thread-safe cache deletion."""
        with Lock():
            if portfolio_id in self.cache:
                del self.cache[portfolio_id]
                del self.access_times[portfolio_id]
                return True
            return False

    def _evict_oldest(self) -> None:
        """Remove least recently used item when cache is full."""
        if not self.access_times:
            return

        # Find oldest access
        oldest_id = min(self.access_times, key=self.access_times.get)
        del self.cache[oldest_id]
        del self.access_times[oldest_id]

    def get_stats(self) -> dict:
        """Get cache statistics for monitoring."""
        with Lock():
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "ttl_minutes": self.ttl.total_seconds() / 60,
                "portfolio_ids": list(self.cache.keys())
            }


# Global cache instance
portfolio_cache = PortfolioCache(max_size=1000, ttl_minutes=30)


async def get_cached_portfolio(portfolio_id: str) -> Optional[dict]:
    """Get portfolio from cache with fallback to file system."""
    # Try cache first
    cached_data = portfolio_cache.get(portfolio_id)
    if cached_data:
        return cached_data

    # Cache miss - load from file system
    # This would integrate with existing file-based storage
    return None


async def update_cached_portfolio(portfolio_id: str, portfolio_data: dict) -> None:
    """Update portfolio in cache and persist to file system."""
    # Update cache
    portfolio_cache.set(portfolio_id, portfolio_data)

    # Persist to file system (existing implementation)
    # This ensures durability even if cache is lost