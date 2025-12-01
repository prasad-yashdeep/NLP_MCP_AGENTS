"""
Calculator Agent for financial calculations.

This specialized agent uses gemma3:4b to perform financial calculations
including portfolio value, returns, and share computations.
"""

import logging
from typing import List, Optional, Dict, Any

from .base import OllamaAgent
from .models import TaskPlan, StockData, CalculationResult
from .tools import MCPToolWrapper, calculate

logger = logging.getLogger(__name__)


CALCULATOR_AGENT_SYSTEM_PROMPT = """You are a specialized Calculator Agent for financial computations.

Your task is to:
1. Calculate the number of shares purchased based on investment amount and stock price
2. Compute final portfolio value based on ending stock prices
3. Calculate percentage returns

Use these formulas:
- shares = investment_amount / start_price
- final_value = shares * end_price
- percentage_return = ((final_value - investment_amount) / investment_amount) * 100

For multiple stocks with allocation splits:
- investment_per_stock = total_investment * split_percentage
- Calculate shares and final value for each stock
- Sum all final values for total portfolio value

Always return precise calculations with appropriate decimal places.
"""


class CalculatorAgent(OllamaAgent):
    """
    Specialized agent for financial calculations.

    Uses gemma3:4b for lightweight inference and can optionally
    connect to a calculator MCP server for precise calculations.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        tool_wrapper: Optional[MCPToolWrapper] = None,
        use_mcp_calculator: bool = False
    ):
        """
        Initialize the Calculator Agent.

        Args:
            model: Ollama model to use (default: gemma3:4b)
            ollama_url: Ollama API URL
            tool_wrapper: Pre-initialized MCPToolWrapper for MCP tools
            use_mcp_calculator: Whether to use MCP calculator for calculations
        """
        super().__init__(
            name="CalculatorAgent",
            model=model,
            system_prompt=CALCULATOR_AGENT_SYSTEM_PROMPT,
            temperature=0.1,  # Very low temperature for calculations
            max_tokens=1024,
            ollama_url=ollama_url
        )
        self._tool_wrapper = tool_wrapper
        self._use_mcp_calculator = use_mcp_calculator
        self._initialized = False

    async def initialize(self, server_configs_path: Optional[str] = None):
        """Initialize MCP connections if using MCP calculator."""
        if self._use_mcp_calculator and self._tool_wrapper is None:
            self._tool_wrapper = MCPToolWrapper(server_configs_path)
            await self._tool_wrapper.initialize(servers=["calculator"])
        self._initialized = True

    async def cleanup(self):
        """Cleanup MCP connections."""
        if self._tool_wrapper and self._use_mcp_calculator:
            await self._tool_wrapper.cleanup()
        self._initialized = False

    async def compute(
        self,
        stock_data: List[StockData],
        investment: float,
        splits: List[float]
    ) -> CalculationResult:
        """
        Compute portfolio value and returns.

        Args:
            stock_data: List of StockData with prices
            investment: Total investment amount
            splits: Allocation percentages (should match stock_data length)

        Returns:
            CalculationResult with total value and percentage return
        """
        if len(stock_data) != len(splits):
            raise ValueError(
                f"Number of stocks ({len(stock_data)}) must match "
                f"number of splits ({len(splits)})"
            )

        # Validate splits sum to 1.0
        if abs(sum(splits) - 1.0) > 0.01:
            logger.warning(f"Splits sum to {sum(splits)}, normalizing...")
            total = sum(splits)
            splits = [s / total for s in splits]

        total_final_value = 0.0
        details = {}

        for stock, split in zip(stock_data, splits):
            # Calculate investment for this stock
            stock_investment = investment * split

            # Calculate shares purchased
            shares = stock_investment / stock.start_price

            # Calculate final value
            final_value = shares * stock.end_price

            total_final_value += final_value

            # Store details
            details[stock.ticker] = {
                "investment": stock_investment,
                "start_price": stock.start_price,
                "end_price": stock.end_price,
                "shares": shares,
                "final_value": final_value,
                "return_pct": ((final_value - stock_investment) / stock_investment) * 100
            }

            logger.info(
                f"{stock.ticker}: ${stock_investment:.2f} -> "
                f"{shares:.4f} shares -> ${final_value:.2f}"
            )

        # Calculate total percentage return
        percentage_return = ((total_final_value - investment) / investment) * 100

        logger.info(
            f"Total: ${investment:.2f} -> ${total_final_value:.2f} "
            f"({percentage_return:.2f}% return)"
        )

        return CalculationResult(
            total_value=total_final_value,
            percentage_return=percentage_return,
            details=details
        )

    async def compute_from_plan(
        self,
        plan: TaskPlan,
        stock_data: List[StockData]
    ) -> CalculationResult:
        """
        Compute calculations based on a TaskPlan.

        Args:
            plan: TaskPlan containing investment and splits
            stock_data: Stock data from Data Agent

        Returns:
            CalculationResult with computed values
        """
        return await self.compute(
            stock_data=stock_data,
            investment=plan.investment,
            splits=plan.splits
        )

    async def _calculate_with_mcp(self, expression: str) -> Optional[float]:
        """Use MCP calculator for precise calculation."""
        if not self._use_mcp_calculator or not self._tool_wrapper:
            return None

        try:
            result = await calculate(self._tool_wrapper, expression)
            return result
        except Exception as e:
            logger.warning(f"MCP calculation failed: {e}")
            return None

    def calculate_shares(self, investment: float, price: float) -> float:
        """Calculate number of shares."""
        return investment / price

    def calculate_value(self, shares: float, price: float) -> float:
        """Calculate value of shares."""
        return shares * price

    def calculate_return(self, initial: float, final: float) -> float:
        """Calculate percentage return."""
        return ((final - initial) / initial) * 100

    def __repr__(self) -> str:
        return f"CalculatorAgent(model='{self.model}', use_mcp={self._use_mcp_calculator})"
