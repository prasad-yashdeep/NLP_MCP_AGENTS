"""
Multi-Agent Benchmark Integration for MCP Universe.

This module provides a custom agent that integrates the multi-agent
financial analysis system with MCP Universe's BenchmarkRunner.
"""

import os
import sys
import json
import asyncio
import logging
from typing import Dict, Union, Optional, List
from dataclasses import dataclass, field

# Add MCP-Universe to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'MCP-Universe'))

from mcpuniverse.agent.base import BaseAgent, BaseAgentConfig
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.tracer import Tracer

from .financial_manager import FinancialAnalysisManager

logger = logging.getLogger(__name__)


@dataclass
class MultiAgentFinancialConfig(BaseAgentConfig):
    """
    Configuration for Multi-Agent Financial Analysis Agent.

    Extends BaseAgentConfig with multi-agent specific settings.
    """
    name: str = "MultiAgent-Financial"
    instruction: str = "Multi-agent system for financial analysis using orchestrator-worker pattern"

    # Model configurations
    orchestrator_model: str = "gemma3:27b"
    worker_model: str = "gemma3:4b"

    # Agent settings
    max_iterations: int = 10

    # MCP servers to use
    servers: List[Dict] = field(default_factory=lambda: [
        {"name": "yfinance"},
        {"name": "calculator"}
    ])


class MultiAgentFinancial(BaseAgent):
    """
    Multi-Agent Financial Analysis Agent for MCP Universe.

    This agent wraps the FinancialAnalysisManager to provide
    compatibility with the MCP Universe BenchmarkRunner.
    """

    config_class = MultiAgentFinancialConfig
    alias = ["multi_agent_financial", "maf"]

    def __init__(
        self,
        mcp_manager: MCPManager | None,
        llm: BaseLLM | None,
        config: Optional[Union[Dict, str]] = None,
    ):
        """
        Initialize the Multi-Agent Financial Analysis Agent.

        Args:
            mcp_manager: MCP server manager
            llm: LLM instance (can be None, multi-agent uses its own LLMs)
            config: Agent configuration
        """
        super().__init__(mcp_manager, llm, config)
        self._manager: Optional[FinancialAnalysisManager] = None
        self._logger = logging.getLogger(f"{self.__class__.__name__}:{self._name}")

    async def _initialize(self):
        """Initialize the multi-agent system."""
        self._logger.info("Initializing Multi-Agent Financial Analysis system...")

        # Get MCP server configs path
        server_configs_path = None
        if self._mcp_manager:
            server_configs_path = self._mcp_manager.get_config_folder()

        # Initialize the financial analysis manager
        self._manager = FinancialAnalysisManager(
            orchestrator_model=self._config.orchestrator_model,
            worker_model=self._config.worker_model,
            server_configs_path=server_configs_path
        )
        await self._manager.initialize()

        self._logger.info("Multi-Agent Financial Analysis system initialized")

    async def _cleanup(self):
        """Cleanup multi-agent resources."""
        if self._manager:
            await self._manager.cleanup()
        self._logger.info("Multi-Agent Financial Analysis system cleaned up")

    async def _execute(
        self,
        message: Union[str, List[str]],
        output_format: Optional[Union[str, Dict]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Execute the multi-agent financial analysis.

        Args:
            message: Input query (financial analysis question)
            output_format: Expected output format
            **kwargs: Additional arguments including tracer and callbacks

        Returns:
            AgentResponse with the analysis result
        """
        if not self._manager:
            raise RuntimeError("Multi-agent system not initialized")

        # Convert message to string if it's a list
        if isinstance(message, list):
            query = " ".join(message)
        else:
            query = message

        tracer = kwargs.get("tracer", Tracer())

        try:
            with tracer.sprout() as t:
                # Run the multi-agent analysis
                result = await self._manager.analyze(query, output_format)

                # Log the execution
                t.add({
                    "type": "multi_agent_financial",
                    "class": self.__class__.__name__,
                    "agent_name": self._config.name,
                    "input": query,
                    "output": json.dumps(result),
                    "orchestrator_model": self._config.orchestrator_model,
                    "worker_model": self._config.worker_model,
                    "error": ""
                })

                # Create response
                response = AgentResponse(
                    name=self._name,
                    class_name=self.__class__.__name__,
                    response=json.dumps(result),
                    trace_id=t.id if hasattr(t, 'id') else ""
                )

                return response

        except Exception as e:
            self._logger.error(f"Multi-agent execution failed: {e}")

            # Log error
            with tracer.sprout() as t:
                t.add({
                    "type": "multi_agent_financial",
                    "class": self.__class__.__name__,
                    "agent_name": self._config.name,
                    "input": query,
                    "output": "",
                    "error": str(e)
                })
            raise

    def get_description(self, with_tools_description=True) -> str:
        """Get agent description."""
        description = self._config.instruction
        text = f"Multi-Agent Financial Analysis: {self._name}\n"
        text += f"Orchestrator: {self._config.orchestrator_model}\n"
        text += f"Workers: {self._config.worker_model}\n"
        text += f"Description: {description}"

        if with_tools_description:
            text += "\nSub-agents: DataAgent, CalculatorAgent, FormatterAgent"
            text += "\nMCP Tools: yfinance, calculator"

        return text


# Register the agent type with MCP Universe
def register_agent():
    """
    Register the MultiAgentFinancial agent with MCP Universe.

    This should be called during module import or initialization.
    """
    try:
        from mcpuniverse.common.misc import ComponentABCMeta
        # The agent is auto-registered via metaclass when imported
        logger.info("MultiAgentFinancial agent registered with MCP Universe")
    except ImportError:
        logger.warning("Could not register agent with MCP Universe")


# Auto-register on import
register_agent()
