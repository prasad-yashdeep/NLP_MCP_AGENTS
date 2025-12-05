"""
Fact Verification Agent.

This agent cross-references and verifies facts from multiple sources.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Union, List

from mcpuniverse.mcp.manager import MCPManager
from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.agent.harmony_agent import HarmonyReActConfig
from mcpuniverse.agent.web_search.base_worker import BaseWebSearchWorker


FACT_VERIFICATION_INSTRUCTION = """
You are a Fact Verification specialist. Cross-reference and verify facts
from multiple sources.

Facts to verify:
{facts}

Existing sources consulted:
{sources}

Instructions:
1. Use google_search to find additional authoritative sources to verify each fact
2. Look for official sources (.gov, .edu), established news outlets, or primary sources
3. Compare information across sources
4. Identify any conflicting information

For each fact, determine:
1. Is it confirmed by multiple sources?
2. Are there any conflicting reports?
3. What is the confidence level (0.0-1.0)?
4. Which sources are most authoritative?

Return your verification results in the final channel as JSON:
{{
    "verified_facts": [
        {{
            "fact": "Original fact statement",
            "verified": true/false,
            "confidence": 0.X,
            "supporting_sources": ["url1", "url2"],
            "notes": "Any additional context"
        }}
    ],
    "conflicts": [
        {{
            "fact": "Fact with conflicting info",
            "conflict": "Description of the conflict",
            "sources": ["source1", "source2"]
        }}
    ]
}}
"""


@dataclass
class FactVerificationConfig(HarmonyReActConfig):
    """Configuration for FactVerificationAgent."""
    instruction: str = "You are a fact verification specialist."
    max_iterations: int = 4
    summarize_tool_response: bool = True


class FactVerificationAgent(BaseWebSearchWorker):
    """
    Worker agent for cross-referencing and verifying facts.
    Uses google_search for verification searches.
    """
    config_class = FactVerificationConfig
    alias = ["fact_verification"]
    worker_name = "FactVerificationAgent"

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
        """Generate instruction for fact verification task."""
        facts = task_params.get("facts", [])
        if isinstance(facts, list):
            facts_str = "\n".join(f"- {f}" for f in facts)
        else:
            facts_str = str(facts)

        sources = task_params.get("sources", [])
        sources_str = "\n".join(f"- {s}" for s in sources) if sources else "None yet"

        return FACT_VERIFICATION_INSTRUCTION.format(
            facts=facts_str,
            sources=sources_str
        )
