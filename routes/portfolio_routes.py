"""FastAPI routes demonstrating portfolio management for 100+ concurrent users."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict
import asyncio

from services.portfolio_manager_service import get_portfolio_service

router = APIRouter(prefix="/portfolio", tags=["portfolio"])


# Request/Response Models
class PortfolioCreate(BaseModel):
    name: str = Field(..., description="Portfolio name")
    initial_capital: float = Field(..., gt=0, description="Initial investment amount")


class PortfolioAddStock(BaseModel):
    portfolio_id: str = Field(..., description="Portfolio ID")
    symbol: str = Field(..., description="Stock symbol")
    shares: float = Field(..., gt=0, description="Number of shares")
    buy_price: float = Field(..., gt=0, description="Purchase price per share")


class PortfolioResponse(BaseModel):
    portfolio_id: str
    name: str
    initial_capital: float
    current_value: float
    holdings: Dict[str, dict]
    created_at: str


@router.post("/create", response_model=Dict[str, str])
async def create_portfolio(request: PortfolioCreate, background_tasks: BackgroundTasks):
    """
    Create new portfolio - handles 100+ concurrent requests efficiently.

    Uses:
    - Async operations for non-blocking I/O
    - Multi-tier caching for performance
    - Background tasks for non-critical operations
    """
    try:
        # Generate unique ID
        import random
        portfolio_id = f"PORT_{random.randint(10000, 99999)}"

        # Create portfolio data
        portfolio_data = {
            "id": portfolio_id,
            "name": request.name,
            "initial_capital": request.initial_capital,
            "current_value": request.initial_capital,
            "holdings": {},
            "created_at": asyncio.get_event_loop().time()
        }

        # Get service with Redis enabled for production
        service = await get_portfolio_service(use_redis=True)

        # Save portfolio (async, non-blocking)
        success = await service.save_portfolio(portfolio_id, portfolio_data)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to create portfolio")

        # Add background task for analytics, logging, etc.
        background_tasks.add_task(log_portfolio_creation, portfolio_id, request.name)

        return {
            "portfolio_id": portfolio_id,
            "name": request.name,
            "status": "created",
            "message": "Portfolio created successfully"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating portfolio: {str(e)}")


@router.post("/add-stock")
async def add_stock_to_portfolio(request: PortfolioAddStock):
    """
    Add stock to portfolio - thread-safe for concurrent access.

    Uses:
    - Optimistic locking for concurrent updates
    - Multi-tier caching strategy
    - Automatic retry on conflicts
    """
    try:
        service = await get_portfolio_service(use_redis=True)

        # Define thread-safe update function
        def add_stock_update(current_portfolio):
            if not current_portfolio:
                raise ValueError("Portfolio not found")

            holdings = current_portfolio.get("holdings", {})
            symbol = request.symbol.upper()

            if symbol in holdings:
                # Update existing position
                existing = holdings[symbol]
                total_shares = existing["shares"] + request.shares
                total_cost = existing["total_cost"] + (request.shares * request.buy_price)
                holdings[symbol] = {
                    "shares": total_shares,
                    "average_cost": total_cost / total_shares,
                    "total_cost": total_cost
                }
            else:
                # Add new position
                holdings[symbol] = {
                    "shares": request.shares,
                    "average_cost": request.buy_price,
                    "total_cost": request.shares * request.buy_price
                }

            current_portfolio["holdings"] = holdings
            return current_portfolio

        # Perform thread-safe update with automatic retries
        updated_portfolio = await service.update_portfolio(
            request.portfolio_id,
            add_stock_update
        )

        if not updated_portfolio:
            raise HTTPException(status_code=404, detail="Portfolio not found or update failed")

        return {
            "status": "success",
            "portfolio_id": request.portfolio_id,
            "symbol": request.symbol,
            "shares_added": request.shares,
            "total_shares": updated_portfolio["holdings"][request.symbol.upper()]["shares"],
            "new_average_cost": updated_portfolio["holdings"][request.symbol.upper()]["average_cost"]
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding stock: {str(e)}")


@router.get("/{portfolio_id}", response_model=PortfolioResponse)
async def get_portfolio(portfolio_id: str):
    """
    Get portfolio details - optimized for high concurrency.

    Uses:
    - Multi-tier caching for sub-millisecond response times
    - Efficient data serialization
    """
    try:
        service = await get_portfolio_service(use_redis=True)
        portfolio_data = await service.get_portfolio(portfolio_id)

        if not portfolio_data:
            raise HTTPException(status_code=404, detail="Portfolio not found")

        return PortfolioResponse(**portfolio_data)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving portfolio: {str(e)}")


@router.get("/stats/cache")
async def get_cache_stats():
    """Get cache statistics - useful for monitoring and optimization."""
    try:
        service = await get_portfolio_service(use_redis=True)
        stats = await service.get_portfolio_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")


@router.post("/warmup")
async def warm_up_cache(portfolio_ids: list[str]):
    """
    Pre-load portfolios into cache - useful for expected traffic patterns.

    Example: After server restart, warm up cache for active users
    """
    try:
        service = await get_portfolio_service(use_redis=True)
        results = await service.warm_up_cache(portfolio_ids)
        return {
            "status": "completed",
            "results": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error warming up cache: {str(e)}")


# Background task for non-critical operations
async def log_portfolio_creation(portfolio_id: str, name: str):
    """Example background task for analytics, logging, etc."""
    # This runs asynchronously without blocking the response
    await asyncio.sleep(0.1)  # Simulate some async work
    print(f"Portfolio created: {portfolio_id} - {name}")
    # Could add database logging, analytics, etc.


# Load testing endpoint
@router.get("/load-test")
async def load_test_concurrent_users():
    """
    Test endpoint to simulate 100 concurrent users.

    In production, you'd use tools like:
    - Apache Bench: ab -n 1000 -c 100 http://localhost:8000/portfolio/load-test
    - Locust: Python load testing framework
    - JMeter: Java-based load testing
    """
    async def simulate_user(user_id: int):
        """Simulate a single user's portfolio operations."""
        try:
            # Create portfolio
            create_request = PortfolioCreate(
                name=f"User {user_id} Portfolio",
                initial_capital=10000.0
            )
            result = await create_portfolio(create_request, BackgroundTasks())
            portfolio_id = result["portfolio_id"]

            # Add stocks
            add_request = PortfolioAddStock(
                portfolio_id=portfolio_id,
                symbol="AAPL",
                shares=10.0,
                buy_price=175.0
            )
            await add_stock_to_portfolio(add_request)

            # Get portfolio
            portfolio = await get_portfolio(portfolio_id)

            return {"user_id": user_id, "status": "success", "portfolio_id": portfolio_id}

        except Exception as e:
            return {"user_id": user_id, "status": "error", "error": str(e)}

    # Simulate 100 concurrent users
    tasks = [simulate_user(i) for i in range(1, 101)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    success_count = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")

    return {
        "total_users": 100,
        "successful_operations": success_count,
        "failed_operations": 100 - success_count,
        "timestamp": asyncio.get_event_loop().time()
    }