"""
Synthesis Agent.

This agent aggregates and formats the final response from all collected data.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union
import json

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.harmony_agent import HarmonyReActConfig
from mcpuniverse.agent.web_search.base_worker import BaseWebSearchWorker


SYNTHESIS_INSTRUCTION = """
You are a Synthesis specialist. Aggregate and format the final response
from all collected data to answer the user's original question.

Original Question: {original_query}

Collected Data:
- Search queries executed: {queries}
- Facts extracted: {facts}
- Sources consulted: {sources}

Required Output Format: {output_format}

Instructions:
1. Review all collected data and facts
2. Create a comprehensive, well-organized answer to the original question
3. Include proper citations for each claim
4. Note any areas of uncertainty or conflicting information
5. Follow the exact output format specified

Do not use any tools. Synthesize the answer from the provided data.

Return your final answer in the final channel, following the required output format.
If no specific format is given, return a JSON object with:
{{
    "answer": "Your comprehensive answer here",
    "key_facts": ["fact1", "fact2", ...],
    "sources": ["source1", "source2", ...],
    "confidence": 0.X,
    "notes": "Any caveats or additional context"
}}
"""


@dataclass
class SynthesisConfig(HarmonyReActConfig):
    """Configuration for SynthesisAgent."""
    instruction: str = "You are a synthesis specialist for creating final answers."
    max_iterations: int = 2
    summarize_tool_response: bool = False


class SynthesisAgent(BaseWebSearchWorker):
    """
    Worker agent for synthesizing final output.
    Does NOT use MCP tools - processes accumulated data.
    """
    config_class = SynthesisConfig
    alias = ["synthesis"]
    worker_name = "SynthesisAgent"

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
        """Generate instruction for synthesis task."""
        # Get data from state
        original_query = ""
        queries = []
        facts = []
        sources = []

        if self._state:
            original_query = self._state.original_query
            queries = self._state.queries
            facts = [f.statement for f in self._state.extracted_facts]
            sources = list(self._state.fetched_content.keys())

            # Also extract facts from SERP results if no explicit facts
            if not facts:
                for query, result in self._state.serp_results.items():
                    if isinstance(result, dict):
                        raw = result.get("raw_result", "")
                        if raw:
                            facts.append(f"From search '{query}': {raw[:200]}...")

        output_format = task_params.get("output_format", "{}")
        if isinstance(output_format, dict):
            output_format = json.dumps(output_format, indent=2)

        return SYNTHESIS_INSTRUCTION.format(
            original_query=original_query,
            queries=queries,
            facts=facts if facts else ["No explicit facts extracted yet"],
            sources=sources if sources else ["No sources fetched yet"],
            output_format=output_format
        )
