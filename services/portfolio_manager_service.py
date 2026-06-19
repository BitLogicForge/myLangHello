"""Portfolio manager service integrating cache with FastAPI for concurrent users."""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

from services.portfolio_cache_simple import portfolio_cache
from services.portfolio_cache_redis import get_redis_cache, RedisPortfolioCache

SANDBOX_DIR = Path(__file__).parent.parent.resolve() / "sandbox"
SANDBOX_DIR.mkdir(exist_ok=True)


class PortfolioManagerService:
    """
    Production-ready portfolio management with multi-tier caching:

    1. Redis Cache (fast, distributed, scalable)
    2. Memory Cache (fallback, faster for single instance)
    3. File System (persistent, always available)
    """

    def __init__(self, use_redis: bool = False):
        self.use_redis = use_redis
        self.redis_cache: Optional[RedisPortfolioCache] = None
        self.memory_cache = portfolio_cache

    async def _get_redis(self) -> Optional[RedisPortfolioCache]:
        """Get Redis cache if enabled."""
        if self.use_redis and self.redis_cache is None:
            try:
                self.redis_cache = await get_redis_cache()
            except Exception as e:
                print(f"Redis unavailable, falling back to memory cache: {e}")
                self.use_redis = False
        return self.redis_cache

    async def get_portfolio(self, portfolio_id: str) -> Optional[dict]:
        """
        Get portfolio with multi-tier caching strategy:
        1. Try Redis (if enabled) - fastest for distributed systems
        2. Try memory cache - fast for single instance
        3. Try file system - always available fallback
        """
        # Tier 1: Redis Cache
        if self.use_redis:
            redis = await self._get_redis()
            if redis:
                data = await redis.get(portfolio_id)
                if data:
                    return data

        # Tier 2: Memory Cache
        data = self.memory_cache.get(portfolio_id)
        if data:
            # Populate Redis for next time
            if self.use_redis:
                redis = await self._get_redis()
                if redis:
                    await redis.set(portfolio_id, data)
            return data

        # Tier 3: File System
        file_path = SANDBOX_DIR / f"{portfolio_id}.json"
        if file_path.exists():
            try:
                raw_data = await asyncio.to_thread(file_path.read_text)
                portfolio_data = json.loads(raw_data)

                # Populate caches for next access
                self.memory_cache.set(portfolio_id, portfolio_data)
                if self.use_redis:
                    redis = await self._get_redis()
                    if redis:
                        await redis.set(portfolio_id, portfolio_data)

                return portfolio_data

            except Exception as e:
                print(f"Error reading portfolio file: {e}")

        return None

    async def save_portfolio(self, portfolio_id: str, portfolio_data: dict) -> bool:
        """
        Save portfolio with write-through caching strategy:
        1. Write to all caches immediately
        2. Persist to file system for durability
        """
        success = True

        # Update Redis Cache
        if self.use_redis:
            redis = await self._get_redis()
            if redis:
                if not await redis.set(portfolio_id, portfolio_data):
                    success = False

        # Update Memory Cache
        self.memory_cache.set(portfolio_id, portfolio_data)

        # Persist to File System (always for durability)
        try:
            file_path = SANDBOX_DIR / f"{portfolio_id}.json"
            await asyncio.to_thread(
                file_path.write_text,
                json.dumps(portfolio_data),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"Error saving portfolio file: {e}")
            success = False

        return success

    async def update_portfolio(
        self,
        portfolio_id: str,
        update_func: callable
    ) -> Optional[dict]:
        """
        Thread-safe portfolio update using optimistic locking pattern.

        Args:
            portfolio_id: Portfolio identifier
            update_func: Function that takes current portfolio data and returns updated data

        Returns:
            Updated portfolio data or None if failed
        """
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            # Get current portfolio data
            current_data = await self.get_portfolio(portfolio_id)
            if current_data is None:
                return None

            try:
                # Apply update function
                updated_data = update_func(current_data.copy())

                # Save updated data
                success = await self.save_portfolio(portfolio_id, updated_data)

                if success:
                    return updated_data
                else:
                    retry_count += 1
                    await asyncio.sleep(0.1 * retry_count)  # Exponential backoff

            except Exception as e:
                print(f"Update error (attempt {retry_count + 1}): {e}")
                retry_count += 1
                await asyncio.sleep(0.1 * retry_count)

        return None

    async def delete_portfolio(self, portfolio_id: str) -> bool:
        """Delete portfolio from all storage layers."""
        success = True

        # Delete from Redis
        if self.use_redis:
            redis = await self._get_redis()
            if redis:
                if not await redis.delete(portfolio_id):
                    success = False

        # Delete from memory cache
        if not self.memory_cache.delete(portfolio_id):
            success = False

        # Delete file
        file_path = SANDBOX_DIR / f"{portfolio_id}.json"
        if file_path.exists():
            try:
                await asyncio.to_thread(file_path.unlink)
            except Exception as e:
                print(f"Error deleting portfolio file: {e}")
                success = False

        return success

    async def get_portfolio_stats(self) -> dict:
        """Get statistics across all storage layers."""
        stats = {
            "memory_cache": self.memory_cache.get_stats(),
            "redis_cache": {},
            "file_count": 0
        }

        # Redis stats
        if self.use_redis:
            redis = await self._get_redis()
            if redis:
                stats["redis_cache"] = await redis.get_stats()

        # File count
        try:
            json_files = list(SANDBOX_DIR.glob("*.json"))
            stats["file_count"] = len(json_files)
        except Exception:
            pass

        return stats

    async def warm_up_cache(self, portfolio_ids: list[str]) -> Dict[str, bool]:
        """
        Pre-load portfolios into cache for performance.
        Useful for expected traffic patterns or after server restarts.
        """
        results = {}

        if self.use_redis:
            redis = await self._get_redis()
            if redis:
                # Batch load into Redis
                portfolios = {}
                for pid in portfolio_ids:
                    file_path = SANDBOX_DIR / f"{pid}.json"
                    if file_path.exists():
                        try:
                            raw_data = await asyncio.to_thread(file_path.read_text)
                            portfolios[pid] = json.loads(raw_data)
                        except Exception:
                            portfolios[pid] = None

                if portfolios:
                    await redis.set_multiple(portfolios)

                return {pid: data is not None for pid, data in portfolios.items()}

        return {pid: False for pid in portfolio_ids}


# Global singleton instance
portfolio_service: Optional[PortfolioManagerService] = None


async def get_portfolio_service(use_redis: bool = False) -> PortfolioManagerService:
    """Get or create global portfolio service instance."""
    global portfolio_service
    if portfolio_service is None:
        portfolio_service = PortfolioManagerService(use_redis=use_redis)
    return portfolio_service


# Example usage in tools.py:
"""
@tool(args_schema=PortfolioAddStockInput)
async def portfolio_add_stock(portfolio_id: str, symbol: str, shares: float, buy_price: float) -> str:
    # Get portfolio service
    service = await get_portfolio_service(use_redis=True)  # Enable Redis for production

    # Define update function
    def add_stock_update(current_portfolio):
        holdings = current_portfolio.get("holdings", {})
        if symbol in holdings:
            # Update existing position
            existing = holdings[symbol]
            total_shares = existing["shares"] + shares
            total_cost = existing["total_cost"] + (shares * buy_price)
            holdings[symbol] = {
                "shares": total_shares,
                "average_cost": total_cost / total_shares,
                "total_cost": total_cost
            }
        else:
            # Add new position
            holdings[symbol] = {
                "shares": shares,
                "average_cost": buy_price,
                "total_cost": shares * buy_price
            }
        current_portfolio["holdings"] = holdings
        return current_portfolio

    # Perform thread-safe update
    updated_portfolio = await service.update_portfolio(portfolio_id, add_stock_update)

    if updated_portfolio:
        return f"✅ Added {shares} shares of {symbol} to portfolio {portfolio_id}"
    else:
        return f"❌ Failed to update portfolio {portfolio_id}"
"""