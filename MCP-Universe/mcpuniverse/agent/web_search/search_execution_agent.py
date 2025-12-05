"""
Search Execution Agent.

This agent executes web searches using the Google Search MCP server.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.harmony_agent import HarmonyReActConfig
from mcpuniverse.agent.web_search.base_worker import BaseWebSearchWorker


SEARCH_EXECUTION_INSTRUCTION = """
You are a Search Execution specialist. Execute the given search query using
the google-search MCP server tools.

Available tools from google-search server:
- google_search: For general web search (most common)
- google_news: For recent news articles
- google_scholar: For academic papers

Query to execute: {query}
Search type: {search_type}

Instructions:
1. Use the appropriate search tool based on the search_type
2. Execute the search with the provided query
3. Extract and organize the results

From the search results, extract:
1. Top organic results (title, link, snippet)
2. Knowledge graph data if available
3. Answer box content if available
4. Related questions for follow-up

Return your findings in the final channel as JSON:
{{
    "organic_results": [
        {{"title": "...", "link": "...", "snippet": "..."}}
    ],
    "knowledge_graph": {{"title": "...", "facts": {{...}}}},
    "answer_box": {{"answer": "...", "source": "..."}},
    "related_questions": ["...", "..."]
}}
"""


@dataclass
class SearchExecutionConfig(HarmonyReActConfig):
    """Configuration for SearchExecutionAgent."""
    instruction: str = "You are a search execution specialist using Google Search MCP."
    max_iterations: int = 3
    summarize_tool_response: bool = True


class SearchExecutionAgent(BaseWebSearchWorker):
    """
    Worker agent for executing searches via Google Search MCP.
    Uses google_search, google_news, google_scholar tools.
    """
    config_class = SearchExecutionConfig
    alias = ["search_execution"]
    worker_name = "SearchExecutionAgent"

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None,
    ):
        # Set default servers for google-search
        if config is None:
            config = {}
        if isinstance(config, dict) and "servers" not in config:
            config["servers"] = [{"name": "google-search"}]
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)

    def get_task_instruction(self, task_params: Dict[str, Any]) -> str:
        """Generate instruction for search execution task."""
        query = task_params.get("query", "")
        search_type = task_params.get("search_type", "web")

        return SEARCH_EXECUTION_INSTRUCTION.format(
            query=query,
            search_type=search_type
        )
