"""
Content Fetch Agent.

This agent fetches full page content from URLs using the Fetch MCP server.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.harmony_agent import HarmonyReActConfig
from mcpuniverse.agent.web_search.base_worker import BaseWebSearchWorker


CONTENT_FETCH_INSTRUCTION = """
You are a Content Fetch specialist. Retrieve full page content from URLs
using the fetch MCP server.

URL to fetch: {url}
Purpose: {purpose}

Available tools from fetch server:
- fetch: Fetch a URL and return its content as markdown

Instructions:
1. Use the fetch tool to retrieve content from the URL
2. The content will be returned as markdown
3. Extract the most relevant information based on the purpose

From the fetched content, extract:
1. Main article/content text relevant to the purpose
2. Key data points, statistics, or facts
3. Publication date if available
4. Author/source information

Return your findings in the final channel as JSON:
{{
    "url": "{url}",
    "title": "Page title",
    "main_content": "Extracted relevant content...",
    "key_facts": ["fact1", "fact2", ...],
    "publication_date": "date if found",
    "source": "author or organization"
}}
"""


@dataclass
class ContentFetchConfig(HarmonyReActConfig):
    """Configuration for ContentFetchAgent."""
    instruction: str = "You are a content retrieval specialist using Fetch MCP."
    max_iterations: int = 3
    summarize_tool_response: bool = True


class ContentFetchAgent(BaseWebSearchWorker):
    """
    Worker agent for fetching full page content via Fetch MCP.
    Uses fetch tool.
    """
    config_class = ContentFetchConfig
    alias = ["content_fetch"]
    worker_name = "ContentFetchAgent"

    def __init__(
        self,
        mcp_manager: MCPManager,
        llm: BaseLLM,
        config: Optional[Union[Dict, str]] = None,
    ):
        # Set default servers for fetch
        if config is None:
            config = {}
        if isinstance(config, dict) and "servers" not in config:
            config["servers"] = [{"name": "fetch"}]
        super().__init__(mcp_manager=mcp_manager, llm=llm, config=config)

    def get_task_instruction(self, task_params: Dict[str, Any]) -> str:
        """Generate instruction for content fetch task."""
        url = task_params.get("url", "")
        purpose = task_params.get("purpose", "Extract relevant information")

        return CONTENT_FETCH_INSTRUCTION.format(
            url=url,
            purpose=purpose
        )
