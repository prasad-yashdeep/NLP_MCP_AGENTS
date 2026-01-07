"""
Web Search Multi-Agent System.

This module provides specialized agents for web search tasks using the HarmonyReAct pattern.
Each agent is designed for a specific role in the search workflow.
"""
from abc import abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass

from mcpuniverse.agent.harmony_agent import HarmonyReAct, HarmonyReActConfig
from mcpuniverse.agent.web_search.state import (
    WebSearchState,
    SearchResult,
    KnowledgeGraph,
    FetchedContent,
    ExtractedFact
)
from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.callbacks.base import (
    CallbackMessage, MessageType, send_message, send_message_async
)
from mcpuniverse.common.logger import get_logger


@dataclass
class WebSearchWorkerConfig(HarmonyReActConfig):
    """Base config for web search worker agents."""
    max_iterations: int = 3  # Workers have fewer iterations than orchestrator
    summarize_tool_response: bool = True


class BaseWebSearchWorker(HarmonyReAct):
    """
    Base class for web search worker agents.
    Extends HarmonyReAct with state integration and handoff logging.
    """
    config_class = WebSearchWorkerConfig
    worker_name: str = "base_worker"

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None,
    ):
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)
        self._state: Optional[WebSearchState] = None
        self._worker_logger = get_logger(f"WebSearchWorker:{self.worker_name}")

    def set_state(self, state: WebSearchState):
        """Inject shared state from orchestrator."""
        self._state = state

    def get_state(self) -> Optional[WebSearchState]:
        """Get the current shared state."""
        return self._state

    @abstractmethod
    def get_task_instruction(self, task_params: Dict[str, Any]) -> str:
        """
        Generate task-specific instruction for this worker.

        Args:
            task_params: Parameters for the task

        Returns:
            Instruction string for the LLM
        """
        pass

    async def execute_task(
        self,
        task_params: Dict[str, Any],
        callbacks: List = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a specific task with handoff logging.

        Args:
            task_params: Parameters for the task
            callbacks: Callback handlers for logging
            **kwargs: Additional arguments (tracer, etc.)

        Returns:
            Task result dictionary with worker name and result
        """
        if callbacks is None:
            callbacks = []

        # Log handoff from orchestrator
        await self._log_handoff_start(task_params, callbacks)

        # Build task instruction
        instruction = self.get_task_instruction(task_params)

        # Execute using HarmonyReAct pattern
        try:
            response = await self.execute(instruction, **kwargs)
            result_str = response.get_response_str()
        except Exception as e:
            self._worker_logger.error(f"Worker {self.worker_name} failed: {e}")
            result_str = f"Error: {str(e)}"

        # Log completion
        await self._log_handoff_complete(result_str, callbacks)

        # Clear history for next task
        self.clear_history()

        return {
            "worker": self.worker_name,
            "result": result_str,
            "trace_id": response.trace_id if hasattr(response, 'trace_id') else None
        }

    async def _log_handoff_start(self, task_params: Dict, callbacks: List):
        """Log when orchestrator hands off to this worker."""
        action = task_params.get("action", "unknown")
        params_preview = {k: v for k, v in task_params.items() if k != "action"}
        params_str = str(params_preview)[:100] + "..." if len(str(params_preview)) > 100 else str(params_preview)

        log_data = {
            "event": "handoff_start",
            "worker": self.worker_name,
            "action": action,
            "params": params_preview
        }

        # Send structured log
        send_message(
            callbacks,
            message=CallbackMessage(
                source=f"WebSearchWorker:{self.worker_name}",
                type=MessageType.LOG,
                data=log_data
            )
        )

        # Send formatted console output
        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source=f"WebSearchWorker:{self.worker_name}",
                type=MessageType.LOG,
                metadata={
                    "event": "plain_text",
                    "data": (
                        f"\n{'='*60}\n"
                        f"[HANDOFF] Orchestrator -> {self.worker_name}\n"
                        f"[ACTION] {action}({params_str})\n"
                        f"{'='*60}\n"
                    )
                }
            )
        )

    async def _log_handoff_complete(self, result: str, callbacks: List):
        """Log when worker completes and returns to orchestrator."""
        result_preview = result[:200] + "..." if len(result) > 200 else result

        log_data = {
            "event": "handoff_complete",
            "worker": self.worker_name,
            "result_preview": result_preview
        }

        # Send structured log
        send_message(
            callbacks,
            message=CallbackMessage(
                source=f"WebSearchWorker:{self.worker_name}",
                type=MessageType.LOG,
                data=log_data
            )
        )

        # Send formatted console output
        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source=f"WebSearchWorker:{self.worker_name}",
                type=MessageType.LOG,
                metadata={
                    "event": "plain_text",
                    "data": (
                        f"\n{'-'*60}\n"
                        f"[COMPLETE] {self.worker_name} -> Orchestrator\n"
                        f"[RESULT] {result_preview}\n"
                        f"{'-'*60}\n"
                    )
                }
            )
        )


# Import worker agents after base class is defined to avoid circular imports
from mcpuniverse.agent.web_search.query_formulation_agent import QueryFormulationAgent
from mcpuniverse.agent.web_search.search_execution_agent import SearchExecutionAgent
from mcpuniverse.agent.web_search.content_fetch_agent import ContentFetchAgent
from mcpuniverse.agent.web_search.fact_verification_agent import FactVerificationAgent
from mcpuniverse.agent.web_search.synthesis_agent import SynthesisAgent

__all__ = [
    # State management
    "WebSearchState",
    "SearchResult",
    "KnowledgeGraph",
    "FetchedContent",
    "ExtractedFact",
    # Base class
    "BaseWebSearchWorker",
    "WebSearchWorkerConfig",
    # Worker agents
    "QueryFormulationAgent",
    "SearchExecutionAgent",
    "ContentFetchAgent",
    "FactVerificationAgent",
    "SynthesisAgent",
]
