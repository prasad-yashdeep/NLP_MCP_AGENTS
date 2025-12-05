"""
Web Search Orchestrator V2 - Multi-agent web search workflow using OpenAI Agent SDK.

This orchestrator uses OpenAI Agent SDK's native function calling to reliably
delegate to specialized worker agents, solving the unreliable LLM-based action
selection problem of the original implementation.

The orchestrator generates the final answer directly (no separate synthesis agent).
"""
import json
import os
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

import openai
from agents import Agent, Runner, ModelSettings
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from agents.tool import FunctionTool
from agents.tool_context import ToolContext

from mcpuniverse.llm.base import BaseLLM
from mcpuniverse.workflows.base import BaseWorkflow
from mcpuniverse.agent.types import AgentResponse
from mcpuniverse.agent.web_search.state import WebSearchState, FetchedContent
from mcpuniverse.agent.web_search.base_worker import BaseWebSearchWorker
from mcpuniverse.tracer import Tracer
from mcpuniverse.callbacks.base import (
    BaseCallback, CallbackMessage, MessageType,
    send_message, send_message_async
)
from mcpuniverse.common.logger import get_logger


# OpenRouter model mapping
OPENROUTER_MODEL_MAP = {
    "GPTOSS20B_OR": "openai/gpt-oss-20b",
    "GPTOSS120B_OR": "openai/gpt-oss-120b",
    "DeepSeekV3_1_OR": "deepseek/deepseek-chat-v3.1",
    "Qwen3Coder_OR": "qwen/qwen3-coder",
    "KimiK2_OR": "moonshotai/kimi-k2",
}

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


ORCHESTRATOR_SYSTEM_PROMPT = """You are an expert Web Search Orchestrator that finds accurate information and delivers clear, concise answers.

## Available Tools
1. **formulate_query(topic)** - Generate optimized search queries
2. **search_web(query, search_type)** - Execute web search (types: web, news, scholar)
3. **fetch_content(url, purpose)** - Retrieve full page content from a URL
4. **verify_facts(facts, sources)** - Cross-reference facts across sources

## Workflow
1. Analyze the question → formulate 1-2 targeted search queries
2. Search → gather relevant results
3. If needed, fetch authoritative sources for deeper info
4. Verify critical facts if accuracy is paramount

## Final Answer Guidelines
When you have gathered sufficient information, provide your final answer directly (do NOT use any tool).

Your final answer MUST be:
- **Concise**: Get to the point quickly. No filler words or unnecessary preamble.
- **Accurate**: Only state facts you found in search results. Cite sources.
- **Well-structured**: Use bullet points for multiple items. Bold key terms.
- **Direct**: Answer the question first, then provide supporting details.

## Answer Format
For factual questions, structure your response as:
```
[Direct answer to the question]

**Key Facts:**
• [Fact 1] (Source: [domain])
• [Fact 2] (Source: [domain])

**Sources:** [list of URLs consulted]
```

For complex questions requiring analysis:
```
[Concise summary answer - 1-2 sentences]

**Details:**
[Brief supporting information organized logically]

**Sources:** [list of URLs]
```

IMPORTANT:
- Do NOT pad your answer. Short questions deserve short answers.
- Do NOT repeat the question back.
- Do NOT say "Based on my search" or similar - just give the answer.
- If you cannot find information, say so clearly and briefly.
"""


class WebSearchOrchestrator(BaseWorkflow):
    """
    Multi-agent orchestrator for web search using OpenAI Agent SDK.

    Uses native function calling to reliably delegate to HarmonyReAct worker agents.
    Each worker handles specific MCP tool interactions (search, fetch, verify).
    The orchestrator generates the final answer directly.
    """
    alias = ["web_search_orchestrator"]

    def __init__(
        self,
        llm: BaseLLM,
        query_agent: BaseWebSearchWorker,
        search_agent: BaseWebSearchWorker,
        fetch_agent: BaseWebSearchWorker,
        verify_agent: BaseWebSearchWorker,
        max_iterations: int = 12,
        model: str = "GPTOSS120B_OR"
    ):
        """
        Initialize the Web Search Orchestrator.

        Args:
            llm: Language model (used by workers, orchestrator uses OpenAI SDK)
            query_agent: Agent for query formulation
            search_agent: Agent for search execution (google-search MCP)
            fetch_agent: Agent for content fetching (fetch MCP)
            verify_agent: Agent for fact verification (google-search MCP)
            max_iterations: Maximum orchestration turns
            model: Model name for orchestrator (OpenRouter format like GPTOSS120B_OR)
        """
        super().__init__()
        self._name = "web_search_orchestrator"
        self._llm = llm
        self._max_iterations = max_iterations
        self._model = model
        self._logger = get_logger(self.__class__.__name__)

        # Worker agents (HarmonyReAct-based)
        self._query_agent = query_agent
        self._search_agent = search_agent
        self._fetch_agent = fetch_agent
        self._verify_agent = verify_agent

        # All agents for initialization/cleanup (exclude None synthesis_agent)
        self._agents = [
            query_agent, search_agent, fetch_agent, verify_agent
        ]

        # State and OpenAI Agent SDK components
        self._state: Optional[WebSearchState] = None
        self._openai_agent: Optional[Agent] = None
        self._orchestrator_tools: List[FunctionTool] = []

    async def initialize(self):
        """Initialize all worker agents and create the OpenAI orchestrator agent."""
        # Initialize worker agents
        for agent in self._agents:
            await agent.initialize()

        # Create FunctionTools for orchestration
        self._orchestrator_tools = self._create_orchestrator_tools()

        # Create the OpenAI Agent for orchestration
        self._openai_agent = self._create_openai_agent()

    async def cleanup(self):
        """Cleanup all worker agents."""
        for agent in self._agents[::-1]:
            await agent.cleanup()
        self._openai_agent = None
        self._orchestrator_tools = []

    def reset(self):
        """Reset orchestrator state."""
        self._state = None
        for agent in self._agents:
            agent.reset()

    def _get_openai_model(self) -> OpenAIChatCompletionsModel:
        """Create OpenAI model configured for OpenRouter."""
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY environment variable is required")

        # Get actual model name from mapping
        actual_model = OPENROUTER_MODEL_MAP.get(self._model, self._model)

        # Create OpenRouter client
        client = openai.AsyncOpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL
        )

        return OpenAIChatCompletionsModel(
            model=actual_model,
            openai_client=client
        )

    def _create_openai_agent(self) -> Agent:
        """Create the OpenAI Agent for orchestration."""
        model = self._get_openai_model()

        return Agent(
            name="web-search-orchestrator",
            instructions=ORCHESTRATOR_SYSTEM_PROMPT,
            tools=self._orchestrator_tools,
            model=model,
            model_settings=ModelSettings(temperature=0.1)
        )

    def _create_orchestrator_tools(self) -> List[FunctionTool]:
        """Create FunctionTools that wrap worker agent calls."""
        tools = []

        # Tool 1: Formulate Query
        tools.append(self._create_function_tool(
            name="formulate_query",
            description="Generate 1-2 optimized search queries for a topic. Use this first to create effective search terms.",
            params_schema={
                "type": "object",
                "properties": {
                    "topic": {
                        "type": "string",
                        "description": "The topic or question to create search queries for"
                    }
                },
                "required": ["topic"]
            },
            handler=self._handle_formulate_query
        ))

        # Tool 2: Search Web
        tools.append(self._create_function_tool(
            name="search_web",
            description="Execute a web search using Google Search. Returns search results with titles, links, and snippets.",
            params_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to execute"
                    },
                    "search_type": {
                        "type": "string",
                        "description": "Type of search: web (default), news (recent articles), or scholar (academic)",
                        "enum": ["web", "news", "scholar"],
                        "default": "web"
                    }
                },
                "required": ["query"]
            },
            handler=self._handle_search_web
        ))

        # Tool 3: Fetch Content
        tools.append(self._create_function_tool(
            name="fetch_content",
            description="Retrieve full page content from a URL. Use for authoritative sources that need deeper analysis.",
            params_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch content from"
                    },
                    "purpose": {
                        "type": "string",
                        "description": "What information you're looking for (helps extract relevant content)"
                    }
                },
                "required": ["url"]
            },
            handler=self._handle_fetch_content
        ))

        # Tool 4: Verify Facts
        tools.append(self._create_function_tool(
            name="verify_facts",
            description="Cross-reference and verify facts from multiple sources. Use for critical claims that need confirmation.",
            params_schema={
                "type": "object",
                "properties": {
                    "facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of facts/claims to verify"
                    },
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of source URLs already consulted"
                    }
                },
                "required": ["facts"]
            },
            handler=self._handle_verify_facts
        ))

        return tools

    def _create_function_tool(
        self,
        name: str,
        description: str,
        params_schema: Dict,
        handler
    ) -> FunctionTool:
        """Create a FunctionTool with the given parameters."""

        async def on_invoke(ctx: ToolContext, input_json: str) -> str:
            try:
                params = json.loads(input_json) if input_json else {}
                return await handler(params)
            except json.JSONDecodeError as e:
                return f"Error parsing parameters: {e}"
            except Exception as e:
                self._logger.error(f"Tool {name} failed: {e}")
                return f"Error: {str(e)}"

        return FunctionTool(
            name=name,
            description=description,
            params_json_schema=params_schema,
            on_invoke_tool=on_invoke,
            strict_json_schema=False
        )

    # Handler methods that delegate to worker agents

    async def _handle_formulate_query(self, params: Dict) -> str:
        """Handle formulate_query tool call."""
        topic = params.get("topic", "")
        self._logger.info(f"[TOOL] formulate_query(topic={topic[:50]}...)")

        result = await self._query_agent.execute_task(
            task_params={"action": "formulate_query", "topic": topic},
            callbacks=self._current_callbacks,
            tracer=self._current_tracer
        )

        # Update state with queries
        result_str = result.get("result", "")
        try:
            if "{" in result_str:
                json_start = result_str.find("{")
                json_end = result_str.rfind("}") + 1
                parsed = json.loads(result_str[json_start:json_end])
                for q in parsed.get("queries", []):
                    self._state.add_query(q)
        except json.JSONDecodeError:
            if topic:
                self._state.add_query(topic)

        return result_str

    async def _handle_search_web(self, params: Dict) -> str:
        """Handle search_web tool call."""
        query = params.get("query", "")
        search_type = params.get("search_type", "web")
        self._logger.info(f"[TOOL] search_web(query={query[:50]}..., type={search_type})")

        result = await self._search_agent.execute_task(
            task_params={"action": "search", "query": query, "search_type": search_type},
            callbacks=self._current_callbacks,
            tracer=self._current_tracer
        )

        # Update state with search results
        result_str = result.get("result", "")
        self._state.add_serp_result(query, {"raw_result": result_str})

        return result_str

    async def _handle_fetch_content(self, params: Dict) -> str:
        """Handle fetch_content tool call."""
        url = params.get("url", "")
        purpose = params.get("purpose", "Extract relevant information")
        self._logger.info(f"[TOOL] fetch_content(url={url[:50]}...)")

        result = await self._fetch_agent.execute_task(
            task_params={"action": "fetch", "url": url, "purpose": purpose},
            callbacks=self._current_callbacks,
            tracer=self._current_tracer
        )

        # Update state with fetched content
        result_str = result.get("result", "")
        self._state.add_fetched_content(url, FetchedContent(
            url=url,
            title="",
            content=result_str
        ))

        return result_str

    async def _handle_verify_facts(self, params: Dict) -> str:
        """Handle verify_facts tool call."""
        facts = params.get("facts", [])
        sources = params.get("sources", [])
        self._logger.info(f"[TOOL] verify_facts(facts_count={len(facts)})")

        result = await self._verify_agent.execute_task(
            task_params={"action": "verify", "facts": facts, "sources": sources},
            callbacks=self._current_callbacks,
            tracer=self._current_tracer
        )

        return result.get("result", "")

    async def execute(
        self,
        message: Union[str, List[str]],
        output_format: Optional[Union[str, Dict]] = None,
        **kwargs
    ) -> AgentResponse:
        """
        Execute the web search workflow.

        Args:
            message: User's search question
            output_format: Desired output format
            **kwargs: Additional arguments (tracer, callbacks)

        Returns:
            AgentResponse with synthesized answer
        """
        tracer = kwargs.get("tracer", Tracer())
        callbacks = kwargs.get("callbacks", [])

        # Store for handler access
        self._current_tracer = tracer
        self._current_callbacks = callbacks
        self._current_output_format = output_format

        with tracer.sprout() as t:
            # Initialize state
            if isinstance(message, list):
                message = "\n".join(message)

            self._state = WebSearchState(original_query=message)

            # Inject state into all workers
            for agent in self._agents:
                agent.set_state(self._state)

            await self._log_orchestrator_start(message, callbacks)

            # Build the input prompt
            format_instruction = ""
            if output_format:
                if isinstance(output_format, dict):
                    format_instruction = f"\n\nRequired output format (JSON):\n{json.dumps(output_format, indent=2)}"
                else:
                    format_instruction = f"\n\nRequired output format: {output_format}"

            input_prompt = f"""Question: {message}

Find accurate information to answer this question, then provide a clear, concise answer.{format_instruction}"""

            try:
                result = await Runner.run(
                    self._openai_agent,
                    input=input_prompt,
                    max_turns=self._max_iterations,
                )

                final_output = str(result.final_output) if result.final_output else ""

                # Log execution details
                t.add({
                    "type": "web_search_orchestrator",
                    "class": self.__class__.__name__,
                    "input": message,
                    "output": final_output,
                    "queries": self._state.queries,
                    "pages_fetched": len(self._state.fetched_content),
                    "error": ""
                })

            except Exception as e:
                self._logger.error(f"Orchestrator execution failed: {e}")
                final_output = f"Error during search: {str(e)}"
                t.add({
                    "type": "web_search_orchestrator",
                    "class": self.__class__.__name__,
                    "input": message,
                    "output": "",
                    "error": str(e)
                })

            await self._log_orchestrator_complete(callbacks)

            return AgentResponse(
                name=self._name,
                class_name=self.__class__.__name__,
                response=final_output,
                trace_id=t.trace_id
            )

    # Logging methods

    async def _log_orchestrator_start(self, query: str, callbacks: List):
        """Log orchestrator workflow start."""
        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source="WebSearchOrchestrator",
                type=MessageType.LOG,
                metadata={"event": "plain_text", "data":
                    f"\n{'='*70}\n"
                    f"[WEB SEARCH ORCHESTRATOR] Starting search\n"
                    f"Query: {query}\n"
                    f"Model: {self._model}\n"
                    f"{'='*70}\n"
                }
            )
        )

    async def _log_orchestrator_complete(self, callbacks: List):
        """Log orchestrator completion."""
        queries_count = len(self._state.queries) if self._state else 0
        pages_count = len(self._state.fetched_content) if self._state else 0

        await send_message_async(
            callbacks,
            message=CallbackMessage(
                source="WebSearchOrchestrator",
                type=MessageType.LOG,
                metadata={"event": "plain_text", "data":
                    f"\n{'='*70}\n"
                    f"[WEB SEARCH ORCHESTRATOR] Search complete\n"
                    f"Queries: {queries_count} | Pages fetched: {pages_count}\n"
                    f"{'='*70}\n"
                }
            )
        )

    def dump_config(self) -> Dict:
        """Dump workflow configuration."""
        return {
            "type": "workflow",
            "class": self.__class__.__name__,
            "name": self._name,
            "max_iterations": self._max_iterations,
            "model": self._model,
            "agents": [agent._name for agent in self._agents],
        }
