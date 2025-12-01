"""
Fetch Agent for retrieving and parsing web content.

This specialized agent uses gemma3:4b to fetch webpage content
via the fetch MCP server.
"""

import logging
from typing import List, Optional

from .base import OllamaAgent
from .models import FetchedContent, SearchResult
from .tools import MCPToolWrapper, fetch_url, fetch_multiple_urls

logger = logging.getLogger(__name__)


FETCH_AGENT_SYSTEM_PROMPT = """You are a specialized Fetch Agent responsible for retrieving web content.

Your task is to:
1. Receive URLs from the orchestrator or search results
2. Fetch webpage content via the fetch MCP server
3. Parse HTML and extract relevant text content
4. Return structured content with text and metadata

Always retrieve content efficiently and handle errors gracefully.
"""


class FetchAgent(OllamaAgent):
    """
    Specialized agent for fetching and parsing web content.

    Uses gemma3:4b for lightweight inference and connects to
    the fetch MCP server to retrieve webpage content.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        tool_wrapper: Optional[MCPToolWrapper] = None
    ):
        """
        Initialize the Fetch Agent.

        Args:
            model: Ollama model to use (default: gemma3:4b)
            ollama_url: Ollama API URL
            tool_wrapper: Pre-initialized MCPToolWrapper for MCP tools
        """
        super().__init__(
            name="FetchAgent",
            model=model,
            system_prompt=FETCH_AGENT_SYSTEM_PROMPT,
            temperature=0.2,  # Lower temperature for content retrieval
            max_tokens=2048,
            ollama_url=ollama_url
        )
        self._tool_wrapper = tool_wrapper
        self._initialized = False

    async def initialize(self, server_configs_path: Optional[str] = None):
        """Initialize MCP connections if not using pre-initialized wrapper."""
        if self._tool_wrapper is None:
            self._tool_wrapper = MCPToolWrapper(server_configs_path)
            await self._tool_wrapper.initialize(servers=["fetch"])
        self._initialized = True

    async def cleanup(self):
        """Cleanup MCP connections."""
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False

    async def fetch(
        self,
        urls: List[str],
        max_length: int = 10000
    ) -> List[FetchedContent]:
        """
        Fetch content from multiple URLs in parallel.

        Args:
            urls: List of URLs to fetch
            max_length: Maximum content length per URL (default: 10000)

        Returns:
            List of FetchedContent objects with webpage content
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Fetching {len(urls)} URLs...")

        # Fetch all URLs in parallel
        fetched_data = await fetch_multiple_urls(
            self._tool_wrapper,
            urls,
            max_length
        )

        # Convert to FetchedContent objects
        results = []
        for data in fetched_data:
            content = FetchedContent(
                url=data["url"],
                content=data["content"],
                title=data.get("title"),
                status=data.get("status", "success"),
                error_message=data.get("error_message")
            )
            results.append(content)

            if content.status == "success":
                logger.info(
                    f"Fetched {len(content.content)} chars from {content.url}"
                )
                # Log a preview of the content for debugging
                preview = content.content[:200].replace('\n', ' ')
                logger.debug(f"  Content preview: {preview}...")
            else:
                logger.warning(
                    f"Failed to fetch {content.url}: {content.error_message}"
                )

        return results

    async def fetch_single(
        self,
        url: str,
        max_length: int = 10000
    ) -> FetchedContent:
        """
        Fetch content from a single URL.

        Args:
            url: URL to fetch
            max_length: Maximum content length

        Returns:
            FetchedContent object with webpage content
        """
        results = await self.fetch([url], max_length)
        return results[0] if results else FetchedContent(
            url=url,
            content="",
            title=None,
            status="error",
            error_message="Fetch failed"
        )

    async def fetch_from_search_results(
        self,
        search_results: List[SearchResult],
        top_n: int = 5,
        max_length: int = 10000
    ) -> List[FetchedContent]:
        """
        Fetch content from top URLs in search results.

        Args:
            search_results: List of SearchResult objects
            top_n: Number of top results to fetch per query
            max_length: Maximum content length per URL

        Returns:
            List of FetchedContent objects
        """
        # Extract top N URLs from each search result
        urls_to_fetch = []
        for search_result in search_results:
            for result in search_result.results[:top_n]:
                url = result.get("url")
                if url and url not in urls_to_fetch:
                    urls_to_fetch.append(url)

        # Limit total URLs to prevent overwhelming the system
        urls_to_fetch = urls_to_fetch[:top_n * len(search_results)]

        logger.info(
            f"Fetching {len(urls_to_fetch)} URLs from {len(search_results)} search results"
        )

        return await self.fetch(urls_to_fetch, max_length)

    def __repr__(self) -> str:
        return f"FetchAgent(model='{self.model}', initialized={self._initialized})"
