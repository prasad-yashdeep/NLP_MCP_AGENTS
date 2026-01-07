"""
Financial Analysis Manager - coordinates the multi-agent pipeline.

This class orchestrates the flow between:
1. Orchestrator Agent (query parsing)
2. Data Agent (price fetching)
3. Calculator Agent (computations)
4. Formatter Agent (output formatting)
"""

import logging
import asyncio
from typing import Optional, Dict, Any

from .orchestrator import OrchestratorAgent
from .data_agent import DataAgent
from .calculator_agent import CalculatorAgent
from .formatter_agent import FormatterAgent
from .models import TaskPlan, StockData, CalculationResult, FinalOutput
from .tools import MCPToolWrapper

logger = logging.getLogger(__name__)


class FinancialAnalysisManager:
    """
    Manager class that orchestrates multi-agent financial analysis.

    This class implements the pipeline:
    1. Orchestrator parses the query -> TaskPlan
    2. Data Agent fetches stock prices -> List[StockData]
    3. Calculator Agent computes returns -> CalculationResult
    4. Formatter Agent formats output -> FinalOutput
    """

    def __init__(
        self,
        orchestrator_model: str = "gemma3:27b",
        worker_model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        server_configs_path: Optional[str] = None
    ):
        """
        Initialize the Financial Analysis Manager.

        Args:
            orchestrator_model: Model for orchestrator (default: gemma3:27b)
            worker_model: Model for sub-agents (default: gemma3:4b)
            ollama_url: Ollama API URL
            server_configs_path: Path to MCP server configurations
        """
        self._ollama_url = ollama_url
        self._server_configs_path = server_configs_path

        # Initialize agents
        self.orchestrator = OrchestratorAgent(
            model=orchestrator_model,
            ollama_url=ollama_url
        )

        # Shared MCP tool wrapper for agents that need MCP access
        self._tool_wrapper: Optional[MCPToolWrapper] = None

        # Sub-agents (initialized with shared tool wrapper later)
        self._worker_model = worker_model
        self.data_agent: Optional[DataAgent] = None
        self.calculator_agent: Optional[CalculatorAgent] = None
        self.formatter_agent: Optional[FormatterAgent] = None

        self._initialized = False

    async def initialize(self):
        """Initialize the manager and all sub-agents."""
        if self._initialized:
            return

        logger.info("Initializing Financial Analysis Manager...")

        # Initialize shared MCP tool wrapper (use default config)
        self._tool_wrapper = MCPToolWrapper()
        await self._tool_wrapper.initialize(servers=["yfinance", "calculator"])

        # Initialize sub-agents with shared tool wrapper
        self.data_agent = DataAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper
        )
        await self.data_agent.initialize()

        self.calculator_agent = CalculatorAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper,
            use_mcp_calculator=False  # Use Python calculations for speed
        )
        await self.calculator_agent.initialize()

        self.formatter_agent = FormatterAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            use_llm_formatting=False  # Use Python formatting for speed
        )

        self._initialized = True
        logger.info("Financial Analysis Manager initialized successfully")

    async def cleanup(self):
        """Cleanup all resources."""
        if self.data_agent:
            await self.data_agent.cleanup()
        if self.calculator_agent:
            await self.calculator_agent.cleanup()
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False
        logger.info("Financial Analysis Manager cleaned up")

    async def analyze(
        self,
        query: str,
        output_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Analyze a financial question and return structured results.

        Args:
            query: User's financial analysis question
            output_format: Expected output format from benchmark

        Returns:
            Dictionary with 'total value' and 'total percentage return'
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Analyzing query: {query[:100]}...")

        # Step 1: Orchestrator parses the query
        logger.info("Step 1: Orchestrator parsing query...")
        task_plan = await self.orchestrator.plan_async(query, output_format)
        logger.info(f"Task Plan: {task_plan}")

        if not task_plan.tickers or task_plan.tickers == ["UNKNOWN"]:
            logger.error("Failed to extract tickers from query")
            return {
                "total value": "0.00",
                "total percentage return": "0.00"
            }

        # Step 2: Data Agent fetches stock prices
        logger.info("Step 2: Data Agent fetching prices...")
        stock_data = await self.data_agent.fetch_from_plan(task_plan)
        logger.info(f"Stock Data: {stock_data}")

        if not stock_data:
            logger.error("Failed to fetch stock data")
            return {
                "total value": "0.00",
                "total percentage return": "0.00"
            }

        # Step 3: Calculator Agent computes returns
        logger.info("Step 3: Calculator Agent computing returns...")
        calculation = await self.calculator_agent.compute_from_plan(task_plan, stock_data)
        logger.info(f"Calculation Result: {calculation}")

        # Step 4: Formatter Agent formats output
        logger.info("Step 4: Formatter Agent formatting output...")
        final_output = await self.formatter_agent.format(calculation, output_format)
        logger.info(f"Final Output: {final_output}")

        # Return in benchmark format
        return final_output.to_benchmark_format()

    async def analyze_with_details(
        self,
        query: str,
        output_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze with full details for debugging.

        Returns all intermediate results along with the final output.
        """
        if not self._initialized:
            await self.initialize()

        # Step 1: Parse query
        task_plan = await self.orchestrator.plan_async(query, output_format)

        # Step 2: Fetch data
        stock_data = await self.data_agent.fetch_from_plan(task_plan)

        # Step 3: Calculate
        calculation = await self.calculator_agent.compute_from_plan(task_plan, stock_data)

        # Step 4: Format
        final_output = await self.formatter_agent.format(calculation, output_format)

        return {
            "query": query,
            "task_plan": task_plan.model_dump(),
            "stock_data": [sd.model_dump() for sd in stock_data],
            "calculation": calculation.model_dump(),
            "final_output": final_output.to_benchmark_format(),
            "success": True
        }

    def __repr__(self) -> str:
        return (
            f"FinancialAnalysisManager("
            f"orchestrator={self.orchestrator.model}, "
            f"workers={self._worker_model}, "
            f"initialized={self._initialized})"
        )


async def run_analysis(query: str) -> Dict[str, str]:
    """
    Convenience function to run a single financial analysis.

    Args:
        query: Financial analysis question

    Returns:
        Dictionary with analysis results
    """
    manager = FinancialAnalysisManager()
    try:
        await manager.initialize()
        result = await manager.analyze(query)
        return result
    finally:
        await manager.cleanup()
