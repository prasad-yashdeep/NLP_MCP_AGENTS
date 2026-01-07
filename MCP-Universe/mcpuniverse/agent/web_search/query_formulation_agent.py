"""
Query Formulation Agent.

This agent specializes in transforming user questions into optimized search queries.
It uses LLM reasoning only (no MCP tools).
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.harmony_agent import HarmonyReActConfig
from mcpuniverse.agent.web_search.base_worker import BaseWebSearchWorker


QUERY_FORMULATION_INSTRUCTION = """
You are a Query Formulation specialist. Your job is to transform user questions
into optimized search queries for web search engines.

Guidelines:
1. Break complex questions into multiple focused search queries
2. Add temporal qualifiers when freshness matters (e.g., "2024", "current", "latest")
3. Use specific keywords rather than natural language
4. Handle ambiguous terms by creating queries for multiple interpretations
5. Prioritize queries that will yield authoritative sources (.gov, .edu, official sites)
6. For numerical data, include terms like "statistics", "data", "official"

Input topic: {topic}
Original user question: {original_query}

Generate 1-3 optimized search queries. Return your answer in the final channel as JSON:
{{"queries": ["query1", "query2", ...]}}

Do not use any tools. Use only your reasoning to generate the queries.
"""


@dataclass
class QueryFormulationConfig(HarmonyReActConfig):
    """Configuration for QueryFormulationAgent."""
    instruction: str = "You are a search query optimization specialist."
    max_iterations: int = 2
    summarize_tool_response: bool = False


class QueryFormulationAgent(BaseWebSearchWorker):
    """
    Worker agent for generating optimized search queries.
    Does NOT use MCP tools directly - uses LLM reasoning only.
    """
    config_class = QueryFormulationConfig
    alias = ["query_formulation"]
    worker_name = "QueryFormulationAgent"

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None,
    ):
        # Ensure we don't try to connect to any MCP servers
        if config is None:
            config = {}
        if isinstance(config, dict):
            config["servers"] = []  # No MCP servers needed
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)

    def get_task_instruction(self, task_params: Dict[str, Any]) -> str:
        """Generate instruction for query formulation task."""
        topic = task_params.get("topic", "")
        original_query = ""
        if self._state:
            original_query = self._state.original_query

        return QUERY_FORMULATION_INSTRUCTION.format(
            topic=topic,
            original_query=original_query
        )
