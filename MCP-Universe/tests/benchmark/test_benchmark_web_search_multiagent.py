"""
Test suite for multi-agent web search system.

This test runs the full multi-agent web search pipeline using the
WebSearchOrchestrator with specialized worker agents.
"""
import unittest
import pytest
from mcpuniverse.tracer.collectors import FileCollector
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks


class TestWebSearchMultiAgent(unittest.IsolatedAsyncioTestCase):
    """Test suite for multi-agent web search system."""

    @pytest.mark.skip(reason="Requires API keys and MCP servers")
    async def test_multiagent_web_search(self):
        """
        Test the full multi-agent web search pipeline.

        This test:
        1. Loads the web_search_multiagent.yaml config
        2. Initializes the WebSearchOrchestrator with all worker agents
        3. Runs the benchmark tasks
        4. Generates a report with evaluation results

        The logs will clearly show agent handoffs in this format:
        [HANDOFF] Orchestrator -> WorkerAgent
        [ACTION] action_name(params)
        [RESULT] Result preview...
        """
        # Set up trace collector for detailed logging
        trace_collector = FileCollector(log_file="log/web_search_multiagent.log")

        # Load the multi-agent benchmark config
        benchmark = BenchmarkRunner("test/web_search_multiagent.yaml")

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
        print('Multi-Agent Web Search Evaluation Results')
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

    @pytest.mark.skip(reason="Requires API keys")
    async def test_individual_workers(self):
        """
        Test individual worker agents in isolation.

        This test validates each worker agent can:
        1. Initialize properly
        2. Accept state injection
        3. Execute tasks with proper handoff logging
        """
        from mcpuniverse.mcp.manager import MCPManager
        from mcpuniverse.llm.openrouter import OpenRouterModel
        from mcpuniverse.agent.web_search import (
            QueryFormulationAgent,
            SearchExecutionAgent,
            WebSearchState
        )
        from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks

        # Initialize components
        llm = OpenRouterModel(config={
            "model_name": "GPTOSS20B_OR",
            "reasoning": "high"
        })
        mcp_manager = MCPManager()
        callbacks = get_vprint_callbacks()

        # Test QueryFormulationAgent
        print("\n" + "=" * 66)
        print("Testing QueryFormulationAgent")
        print("=" * 66)

        query_agent = QueryFormulationAgent(
            mcp_manager=mcp_manager,
            llm=llm
        )
        await query_agent.initialize()

        state = WebSearchState(original_query="What is the population of Tokyo?")
        query_agent.set_state(state)

        result = await query_agent.execute_task(
            task_params={"action": "formulate_query", "topic": "Tokyo population"},
            callbacks=callbacks
        )

        print(f"\nQuery Agent Result: {result}")

        await query_agent.cleanup()

    async def test_orchestrator_only(self):
        """
        Test the WebSearchOrchestrator with a single query.

        This test validates the full orchestration loop with handoff logging.
        Uses OpenAI Agent SDK for reliable orchestration with native function calling.
        """
        from mcpuniverse.mcp.manager import MCPManager
        from mcpuniverse.llm.openrouter import OpenRouterModel
        from mcpuniverse.workflows import WebSearchOrchestrator
        from mcpuniverse.agent.web_search import (
            QueryFormulationAgent,
            SearchExecutionAgent,
            ContentFetchAgent,
            FactVerificationAgent,
            SynthesisAgent
        )
        from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks
        from mcpuniverse.tracer.collectors import MemoryCollector
        from mcpuniverse.tracer import Tracer

        # Initialize LLM
        llm = OpenRouterModel(config={
            "model_name": "GPTOSS20B_OR",
            "reasoning": "high"
        })

        # Initialize MCP manager
        mcp_manager = MCPManager()

        # Create worker agents
        query_agent = QueryFormulationAgent(mcp_manager=mcp_manager, llm=llm)
        search_agent = SearchExecutionAgent(mcp_manager=mcp_manager, llm=llm)
        fetch_agent = ContentFetchAgent(mcp_manager=mcp_manager, llm=llm)
        verify_agent = FactVerificationAgent(mcp_manager=mcp_manager, llm=llm)
        synthesis_agent = SynthesisAgent(mcp_manager=mcp_manager, llm=llm)

        # Create orchestrator with OpenAI Agent SDK
        orchestrator = WebSearchOrchestrator(
            llm=llm,
            query_agent=query_agent,
            search_agent=search_agent,
            fetch_agent=fetch_agent,
            verify_agent=verify_agent,
            synthesis_agent=synthesis_agent,
            max_iterations=12,
            model="GPTOSS20B_OR"  # Model for OpenAI Agent SDK orchestration
        )

        # Initialize
        await orchestrator.initialize()

        # Set up tracing and callbacks
        collector = MemoryCollector()
        tracer = Tracer(collector=collector)
        callbacks = get_vprint_callbacks()

        # Execute
        print("\n" + "=" * 66)
        print("Testing WebSearchOrchestrator")
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
