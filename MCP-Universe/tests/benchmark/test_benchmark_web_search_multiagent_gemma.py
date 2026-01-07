"""
Test suite for multi-agent web search system using Gemma model.

This test runs the full multi-agent web search pipeline using the
WebSearchOrchestrator with specialized worker agents and Gemma 3 12B model.
"""
import unittest
import pytest
from mcpuniverse.tracer.collectors import FileCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks


class TestWebSearchMultiAgentGemma(unittest.IsolatedAsyncioTestCase):
    """Test suite for multi-agent web search system with Gemma model."""

    async def test_multiagent_web_search_gemma(self):
        """
        Test the full multi-agent web search pipeline with Gemma model.

        This test:
        1. Loads the web_search_multiagent_gemma.yaml config
        2. Initializes the WebSearchOrchestrator with all worker agents
        3. Runs the benchmark tasks
        4. Generates a report with evaluation results

        The logs will clearly show agent handoffs in this format:
        [HANDOFF] Orchestrator -> WorkerAgent
        [ACTION] action_name(params)
        [RESULT] Result preview...
        """
        # Set up trace collector for detailed logging
        trace_collector = FileCollector(log_file="log/web_search_multiagent_gemma.log")

        # Load the multi-agent benchmark config with Gemma model
        benchmark = BenchmarkRunner("test/web_search_multiagent_gemma.yaml")

        # Run the benchmark with verbose callbacks
        results = await benchmark.run(
            trace_collector=trace_collector,
            callbacks=get_vprint_callbacks()
        )

        # Generate report
        report = BenchmarkReport(benchmark, trace_collector=trace_collector)
        report.dump()

        # Print evaluation results
        print('=' * 66)
        print('Multi-Agent Web Search (Gemma) Evaluation Results')
        print('-' * 66)

        for task_name in results[0].task_results.keys():
            print(f"\nTask: {task_name}")
            print('-' * 66)
            eval_results = results[0].task_results[task_name]["evaluation_results"]

            passed_count = sum(1 for r in eval_results if r.passed)
            total_count = len(eval_results)

            print(f"Passed: {passed_count}/{total_count}")

            for eval_result in eval_results:
                status = "\033[32mPASS\033[0m" if eval_result.passed else "\033[31mFAIL\033[0m"
                print(f"  - {eval_result.config.func}: {status}")
                if not eval_result.passed:
                    print(f"    Expected: {eval_result.config.value}")
                    print(f"    Op: {eval_result.config.op}")

    async def test_orchestrator_gemma(self):
        """
        Test the WebSearchOrchestrator with Gemma model.

        This test validates the full orchestration loop with handoff logging.
        Uses OpenAI Agent SDK for orchestration with Gemma 3 12B model.
        """
        from mcpuniverse.mcp.manager import MCPManager
        from mcpuniverse.llm.openrouter import OpenRouterModel
        from mcpuniverse.workflows import WebSearchOrchestrator
        from mcpuniverse.agent.web_search import (
            QueryFormulationAgent,
            SearchExecutionAgent,
            ContentFetchAgent,
            FactVerificationAgent,
        )
        from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks
        from mcpuniverse.tracer.collectors import MemoryCollector
        from mcpuniverse.tracer import Tracer

        # Initialize LLM with Gemma model
        llm = OpenRouterModel(config={
            "model_name": "google/gemma-3-12b-it:free"
        })

        # Initialize MCP manager
        mcp_manager = MCPManager()

        # Create worker agents
        query_agent = QueryFormulationAgent(mcp_manager=mcp_manager, llm=llm)
        search_agent = SearchExecutionAgent(mcp_manager=mcp_manager, llm=llm)
        fetch_agent = ContentFetchAgent(mcp_manager=mcp_manager, llm=llm)
        verify_agent = FactVerificationAgent(mcp_manager=mcp_manager, llm=llm)

        # Create orchestrator with Gemma model
        orchestrator = WebSearchOrchestrator(
            llm=llm,
            query_agent=query_agent,
            search_agent=search_agent,
            fetch_agent=fetch_agent,
            verify_agent=verify_agent,
            max_iterations=12,
            model="google/gemma-3-12b-it:free"
        )

        # Initialize
        await orchestrator.initialize()

        # Set up tracing and callbacks
        collector = MemoryCollector()
        tracer = Tracer(collector=collector)
        callbacks = get_vprint_callbacks()

        # Execute
        print("\n" + "=" * 66)
        print("Testing WebSearchOrchestrator with Gemma Model")
        print("=" * 66)

        response = await orchestrator.execute(
            message="What is the current population of Tokyo, Japan?",
            output_format={
                "city": "<City name>",
                "population": "<Population number>",
                "year": "<Data year>",
                "source": "<Source of information>"
            },
            tracer=tracer,
            callbacks=callbacks
        )

        print("\n" + "=" * 66)
        print("Final Response:")
        print("-" * 66)
        print(response.get_response_str())
        print("=" * 66)

        # Cleanup
        await orchestrator.cleanup()


if __name__ == "__main__":
    unittest.main()
