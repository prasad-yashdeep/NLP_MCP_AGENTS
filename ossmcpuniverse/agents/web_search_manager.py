"""
Web Search Manager - coordinates the multi-agent web search pipeline.

This class orchestrates the flow between:
1. Orchestrator Agent (query decomposition and planning)
2. Search Agent (executing Google searches)
3. Fetch Agent (retrieving webpage content)
4. Synthesis Agent (extracting info and generating answers)
"""

import logging
import asyncio
from typing import Optional, Dict, Any

from .web_search_orchestrator import WebSearchOrchestrator
from .search_agent import SearchAgent
from .fetch_agent import FetchAgent
from .synthesis_agent import SynthesisAgent
from .models import SearchPlan, SearchResult, FetchedContent, ExtractedInfo, WebSearchOutput
from .tools import MCPToolWrapper

logger = logging.getLogger(__name__)


class WebSearchManager:
    """
    Manager class that orchestrates multi-agent web search.

    This class implements the pipeline:
    1. Orchestrator decomposes query -> SearchPlan
    2. Search Agent executes searches -> List[SearchResult]
    3. Fetch Agent retrieves webpages -> List[FetchedContent]
    4. Synthesis Agent extracts info and generates answer -> WebSearchOutput
    """

    def __init__(
        self,
        orchestrator_model: str = "gemma3:27b",
        worker_model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        server_configs_path: Optional[str] = None,
        max_search_iterations: int = 3,
        top_n_urls: int = 5
    ):
        """
        Initialize the Web Search Manager.

        Args:
            orchestrator_model: Model for orchestrator (default: gemma3:27b)
            worker_model: Model for sub-agents (default: gemma3:4b)
            ollama_url: Ollama API URL
            server_configs_path: Path to MCP server configurations
            max_search_iterations: Maximum number of search iterations
            top_n_urls: Number of top URLs to fetch per search
        """
        self._ollama_url = ollama_url
        self._server_configs_path = server_configs_path
        self._max_search_iterations = max_search_iterations
        self._top_n_urls = top_n_urls

        # Initialize orchestrator
        self.orchestrator = WebSearchOrchestrator(
            model=orchestrator_model,
            ollama_url=ollama_url
        )

        # Shared MCP tool wrapper for agents that need MCP access
        self._tool_wrapper: Optional[MCPToolWrapper] = None

        # Sub-agents (initialized with shared tool wrapper later)
        self._worker_model = worker_model
        self.search_agent: Optional[SearchAgent] = None
        self.fetch_agent: Optional[FetchAgent] = None
        self.synthesis_agent: Optional[SynthesisAgent] = None

        self._initialized = False

    async def initialize(self):
        """Initialize the manager and all sub-agents."""
        if self._initialized:
            return

        logger.info("Initializing Web Search Manager...")

        # Initialize shared MCP tool wrapper
        self._tool_wrapper = MCPToolWrapper()
        await self._tool_wrapper.initialize(servers=["google-search", "fetch"])

        # Initialize sub-agents with shared tool wrapper
        self.search_agent = SearchAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper
        )
        await self.search_agent.initialize()

        self.fetch_agent = FetchAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url,
            tool_wrapper=self._tool_wrapper
        )
        await self.fetch_agent.initialize()

        self.synthesis_agent = SynthesisAgent(
            model=self._worker_model,
            ollama_url=self._ollama_url
        )

        self._initialized = True
        logger.info("Web Search Manager initialized successfully")

    async def cleanup(self):
        """Cleanup all resources."""
        if self.search_agent:
            await self.search_agent.cleanup()
        if self.fetch_agent:
            await self.fetch_agent.cleanup()
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False
        logger.info("Web Search Manager cleaned up")

    async def search(
        self,
        query: str,
        output_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Execute a web search to answer the user's question.

        Args:
            query: User's information retrieval question
            output_format: Expected output format from benchmark

        Returns:
            Dictionary with 'answer' field
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"Searching for: {query[:100]}...")

        # Step 1: Orchestrator plans search strategy
        logger.info("Step 1: Orchestrator planning search...")
        search_plan = await self.orchestrator.plan_async(query)
        logger.info(f"Search Plan: {len(search_plan.search_queries)} queries")

        # Step 2: Search Agent executes searches
        logger.info("Step 2: Search Agent executing searches...")
        search_results = await self.search_agent.search(
            search_plan.search_queries,
            num_results=10
        )
        total_results = sum(sr.total_results for sr in search_results)
        logger.info(f"Search Results: {total_results} total results")

        if total_results == 0:
            logger.error("No search results found")
            return {"answer": "No information found"}

        # Step 3: Fetch Agent retrieves top URLs
        logger.info("Step 3: Fetch Agent retrieving content...")
        fetched_content = await self.fetch_agent.fetch_from_search_results(
            search_results,
            top_n=self._top_n_urls,
            max_length=10000
        )
        successful_fetches = sum(1 for fc in fetched_content if fc.status == "success")
        logger.info(
            f"Fetched Content: {successful_fetches}/{len(fetched_content)} successful"
        )

        if successful_fetches == 0:
            logger.warning("Failed to fetch any content, using search snippets only")

        # Step 4: Synthesis Agent generates answer
        logger.info("Step 4: Synthesis Agent extracting info...")
        extracted_info = await self.synthesis_agent.synthesize(
            question=query,
            search_results=search_results,
            fetched_content=fetched_content,
            expected_answer_type=search_plan.expected_answer_type
        )
        logger.info(f"Extracted: {len(extracted_info.facts)} facts")

        # Step 5: Generate final answer
        logger.info("Step 5: Generating final answer...")
        final_output = await self.synthesis_agent.generate_answer(
            question=query,
            extracted_info=extracted_info
        )
        logger.info(f"Final Answer: {final_output.answer}")

        # Return in benchmark format
        return final_output.to_benchmark_format()

    async def search_with_details(
        self,
        query: str,
        output_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Search with full details for debugging.

        Returns all intermediate results along with the final output.
        """
        if not self._initialized:
            await self.initialize()

        # Step 1: Plan search
        search_plan = await self.orchestrator.plan_async(query)

        # Step 2: Execute searches
        search_results = await self.search_agent.search(
            search_plan.search_queries,
            num_results=10
        )

        # Step 3: Fetch content
        fetched_content = await self.fetch_agent.fetch_from_search_results(
            search_results,
            top_n=self._top_n_urls
        )

        # Step 4: Synthesize
        extracted_info = await self.synthesis_agent.synthesize(
            question=query,
            search_results=search_results,
            fetched_content=fetched_content,
            expected_answer_type=search_plan.expected_answer_type
        )

        # Step 5: Generate answer
        final_output = await self.synthesis_agent.generate_answer(
            question=query,
            extracted_info=extracted_info
        )

        return {
            "query": query,
            "search_plan": search_plan.model_dump(),
            "search_results": [sr.model_dump() for sr in search_results],
            "fetched_content": [fc.model_dump() for fc in fetched_content],
            "extracted_info": extracted_info.model_dump(),
            "final_output": final_output.to_benchmark_format(),
            "success": True
        }

    def __repr__(self) -> str:
        return (
            f"WebSearchManager("
            f"orchestrator={self.orchestrator.model}, "
            f"workers={self._worker_model}, "
            f"initialized={self._initialized})"
        )


async def run_web_search(query: str) -> Dict[str, str]:
    """
    Convenience function to run a single web search.

    Args:
        query: Information retrieval question

    Returns:
        Dictionary with search results
    """
    manager = WebSearchManager()
    try:
        await manager.initialize()
        result = await manager.search(query)
        return result
    finally:
        await manager.cleanup()
