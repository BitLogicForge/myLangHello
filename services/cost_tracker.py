"""Cost tracking for LLM API calls."""

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class CostTracker:
    """Track and estimate costs for LLM API usage."""

    # Pricing per 1K tokens (as of 2026 - update as needed)
    PRICING: Dict[str, Dict[str, float]] = {
        # OpenAI pricing
        "gpt-4": {"input": 0.03, "output": 0.06},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4-turbo-preview": {"input": 0.01, "output": 0.03},
        "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
        "gpt-3.5-turbo-16k": {"input": 0.003, "output": 0.004},
        # Azure OpenAI uses same models, prices may vary by region
        # Add specific Azure pricing if needed
        "azure-gpt-4": {"input": 0.03, "output": 0.06},
        "azure-gpt-35-turbo": {"input": 0.0015, "output": 0.002},
    }

    def __init__(self):
        """Initialize cost tracker."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        logger.info("Cost tracker initialized")

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Calculate cost for a single LLM call.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost in dollars
        """
        # Normalize model name
        model_key = model.lower()
        if "gpt-4" in model_key:
            if "turbo" in model_key or "preview" in model_key:
                pricing_key = "gpt-4-turbo"
            else:
                pricing_key = "gpt-4"
        elif "gpt-3.5" in model_key or "gpt-35" in model_key:
            pricing_key = "gpt-3.5-turbo"
        else:
            # Default to GPT-4 pricing if unknown
            logger.warning(f"Unknown model '{model}', using GPT-4 pricing as default")
            pricing_key = "gpt-4"

        pricing = self.PRICING.get(pricing_key, self.PRICING["gpt-4"])

        # Calculate cost (pricing is per 1K tokens)
        input_cost = (input_tokens / 1000) * pricing["input"]
        output_cost = (output_tokens / 1000) * pricing["output"]
        total_cost = input_cost + output_cost

        return round(total_cost, 6)

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Track a single LLM call and update totals.

        Args:
            model: Model identifier
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Cost for this call in dollars
        """
        cost = self.calculate_cost(model, input_tokens, output_tokens)

        self.total_cost += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.call_count += 1

        logger.debug(
            f"LLM call cost: ${cost:.6f} " f"({input_tokens} input + {output_tokens} output tokens)"
        )

        return cost

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cost tracking statistics.

        Returns:
            Dictionary with cost statistics
        """
        return {
            "total_cost": round(self.total_cost, 4),
            "total_calls": self.call_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "avg_cost_per_call": (
                round(self.total_cost / self.call_count, 4) if self.call_count > 0 else 0.0
            ),
        }

    def reset(self) -> None:
        """Reset all counters."""
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.call_count = 0
        logger.info("Cost tracker reset")

    @staticmethod
    def estimate_monthly_cost(
        requests_per_day: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
        model: str = "gpt-4",
    ) -> float:
        """
        Estimate monthly cost based on usage patterns.

        Args:
            requests_per_day: Average number of requests per day
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request
            model: Model to use for estimation

        Returns:
            Estimated monthly cost in dollars
        """
        tracker = CostTracker()
        cost_per_request = tracker.calculate_cost(model, avg_input_tokens, avg_output_tokens)
        monthly_cost = cost_per_request * requests_per_day * 30
        return round(monthly_cost, 2)
