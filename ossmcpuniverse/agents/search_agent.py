"""
Search Agent for executing web searches.

This specialized agent uses gemma3:4b to execute Google search queries
via the google-search MCP server.
"""

import logging
from typing import List, Optional

from .base import OllamaAgent
from .models import SearchResult
from .tools import MCPToolWrapper, google_search

logger = logging.getLogger(__name__)


SEARCH_AGENT_SYSTEM_PROMPT = """You are a specialized Search Agent responsible for executing web searches.

Your task is to:
1. Receive search queries from the orchestrator
2. Execute Google searches via the google-search MCP server
3. Return structured search results with titles, URLs, and snippets

Always execute searches accurately and return all relevant results.
"""


class SearchAgent(OllamaAgent):
    """
    Specialized agent for web search using google-search MCP.

    Uses gemma3:4b for lightweight inference and connects to
    the google-search MCP server to execute searches.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        tool_wrapper: Optional[MCPToolWrapper] = None
    ):
        """
        Initialize the Search Agent.

        Args:
            model: Ollama model to use (default: gemma3:4b)
            ollama_url: Ollama API URL
            tool_wrapper: Pre-initialized MCPToolWrapper for MCP tools
        """
        super().__init__(
            name="SearchAgent",
            model=model,
            system_prompt=SEARCH_AGENT_SYSTEM_PROMPT,
            temperature=0.2,  # Lower temperature for search query execution
            max_tokens=2048,
            ollama_url=ollama_url
        )
        self._tool_wrapper = tool_wrapper
        self._initialized = False

    async def initialize(self, server_configs_path: Optional[str] = None):
        """Initialize MCP connections if not using pre-initialized wrapper."""
        if self._tool_wrapper is None:
            self._tool_wrapper = MCPToolWrapper(server_configs_path)
            await self._tool_wrapper.initialize(servers=["google-search"])
        self._initialized = True

    async def cleanup(self):
        """Cleanup MCP connections."""
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False

    async def search(
        self,
        queries: List[str],
        num_results: int = 10
    ) -> List[SearchResult]:
        """
        Execute multiple search queries.

        Args:
            queries: List of search query strings
            num_results: Number of results per query (default: 10)

        Returns:
            List of SearchResult objects with search results
        """
        if not self._initialized:
            await self.initialize()

        results = []

        for query in queries:
            try:
                logger.info(f"Executing search: {query}")

                # Execute search using MCP tool
                search_results = await google_search(
                    self._tool_wrapper,
                    query,
                    num_results
                )

                # Create SearchResult object
                search_result = SearchResult(
                    query=query,
                    results=search_results,
                    total_results=len(search_results)
                )
                results.append(search_result)

                logger.info(
                    f"Found {len(search_results)} results for '{query}'"
                )

            except Exception as e:
                logger.error(f"Search failed for '{query}': {e}")
                # Add empty result to maintain list structure
                results.append(SearchResult(
                    query=query,
                    results=[],
                    total_results=0
                ))

        return results

    async def search_single(
        self,
        query: str,
        num_results: int = 10
    ) -> SearchResult:
        """
        Execute a single search query.

        Args:
            query: Search query string
            num_results: Number of results to return

        Returns:
            SearchResult object with search results
        """
        results = await self.search([query], num_results)
        return results[0] if results else SearchResult(
            query=query,
            results=[],
            total_results=0
        )

    def __repr__(self) -> str:
        return f"SearchAgent(model='{self.model}', initialized={self._initialized})"
