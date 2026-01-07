"""
Data Agent for fetching financial data.

This specialized agent uses gemma3:4b to fetch stock price data
via the yfinance MCP server.
"""

import logging
from typing import List, Optional, Dict, Any

from .base import OllamaAgent
from .models import TaskPlan, StockData
from .tools import MCPToolWrapper, get_stock_price, get_stock_data_range

logger = logging.getLogger(__name__)


DATA_AGENT_SYSTEM_PROMPT = """You are a specialized Data Agent responsible for fetching financial data.

Your task is to:
1. Receive stock ticker symbols and date ranges
2. Use the yfinance MCP server to fetch accurate stock prices
3. Return structured data with prices for the requested dates

Always return data in a structured format. If a price is not available for a specific date,
try the next available trading day.
"""


class DataAgent(OllamaAgent):
    """
    Specialized agent for fetching financial data.

    Uses gemma3:4b for lightweight inference and connects to
    the yfinance MCP server to retrieve stock prices.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        tool_wrapper: Optional[MCPToolWrapper] = None
    ):
        """
        Initialize the Data Agent.

        Args:
            model: Ollama model to use (default: gemma3:4b)
            ollama_url: Ollama API URL
            tool_wrapper: Pre-initialized MCPToolWrapper for MCP tools
        """
        super().__init__(
            name="DataAgent",
            model=model,
            system_prompt=DATA_AGENT_SYSTEM_PROMPT,
            temperature=0.2,  # Lower temperature for data retrieval
            max_tokens=1024,
            ollama_url=ollama_url
        )
        self._tool_wrapper = tool_wrapper
        self._initialized = False

    async def initialize(self, server_configs_path: Optional[str] = None):
        """Initialize MCP connections if not using pre-initialized wrapper."""
        if self._tool_wrapper is None:
            self._tool_wrapper = MCPToolWrapper(server_configs_path)
            await self._tool_wrapper.initialize(servers=["yfinance"])
        self._initialized = True

    async def cleanup(self):
        """Cleanup MCP connections."""
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False

    async def fetch(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str
    ) -> List[StockData]:
        """
        Fetch stock price data for multiple tickers.

        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            List of StockData objects with price information
        """
        if not self._initialized:
            await self.initialize()

        results = []

        for ticker in tickers:
            try:
                # Fetch prices using MCP tool
                data = await get_stock_data_range(
                    self._tool_wrapper,
                    ticker,
                    start_date,
                    end_date
                )

                if data["start_price"] is not None and data["end_price"] is not None:
                    stock_data = StockData(
                        ticker=ticker,
                        start_price=data["start_price"],
                        end_price=data["end_price"],
                        start_date=start_date,
                        end_date=end_date
                    )
                    results.append(stock_data)
                    logger.info(
                        f"Fetched {ticker}: ${data['start_price']:.2f} -> ${data['end_price']:.2f}"
                    )
                else:
                    # Use LLM to help parse or handle edge cases
                    logger.warning(f"Could not fetch prices for {ticker}, attempting LLM fallback")
                    stock_data = await self._fetch_with_llm_assistance(
                        ticker, start_date, end_date
                    )
                    if stock_data:
                        results.append(stock_data)

            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                # Try LLM fallback
                stock_data = await self._fetch_with_llm_assistance(
                    ticker, start_date, end_date
                )
                if stock_data:
                    results.append(stock_data)

        return results

    async def fetch_from_plan(self, plan: TaskPlan) -> List[StockData]:
        """
        Fetch stock data based on a TaskPlan.

        Args:
            plan: TaskPlan containing tickers and dates

        Returns:
            List of StockData objects
        """
        return await self.fetch(
            tickers=plan.tickers,
            start_date=plan.start_date,
            end_date=plan.end_date
        )

    async def _fetch_with_llm_assistance(
        self,
        ticker: str,
        start_date: str,
        end_date: str
    ) -> Optional[StockData]:
        """
        Use LLM to help interpret MCP tool responses or handle edge cases.

        This is a fallback when direct price parsing fails.
        """
        try:
            # Get raw response from MCP using get_historical_stock_prices
            if self._tool_wrapper:
                raw_response = await self._tool_wrapper.call_tool(
                    "yfinance",
                    "get_historical_stock_prices",
                    {
                        "ticker": ticker,
                        "start_date": start_date,
                        "end_date": end_date,
                        "interval": "1d"
                    }
                )

                # Ask LLM to extract prices
                prompt = f"""Extract the stock closing prices from this historical data.

Ticker: {ticker}
Start date: {start_date}
End date: {end_date}

Historical data:
{raw_response[:2000]}

Find the Close price for dates closest to {start_date} and {end_date}.

Return a JSON object with:
{{"start_price": <number>, "end_price": <number>}}

Only return the JSON, nothing else."""

                response = self.generate(prompt, response_format=None)

                # Parse LLM response
                import json
                import re

                # Try to extract JSON
                json_match = re.search(r'\{[^}]+\}', response)
                if json_match:
                    data = json.loads(json_match.group())
                    return StockData(
                        ticker=ticker,
                        start_price=float(data["start_price"]),
                        end_price=float(data["end_price"]),
                        start_date=start_date,
                        end_date=end_date
                    )

        except Exception as e:
            logger.error(f"LLM fallback failed for {ticker}: {e}")

        return None

    def __repr__(self) -> str:
        return f"DataAgent(model='{self.model}', initialized={self._initialized})"
