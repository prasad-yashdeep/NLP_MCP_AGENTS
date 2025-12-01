"""
ReAct-style Web Search Agent.

This module implements a ReAct (Reasoning + Acting) agent for web search tasks,
following the pattern from MCP-Universe/mcpuniverse/agent/react.py.

The agent follows the Thought-Action-Observation loop:
1. Thought: Reason about what information is needed
2. Action: Decide whether to search, verify, or answer
3. Observation: Process results and update understanding
4. Repeat until confident answer is found
"""

import logging
import json
import re
from typing import Optional, Dict, List

from .base import OllamaAgent
from .models import SearchPlan, WebSearchOutput
from .search_agent import SearchAgent
from .fetch_agent import FetchAgent
from .synthesis_agent import SynthesisAgent
from .web_search_orchestrator import WebSearchOrchestrator

logger = logging.getLogger(__name__)


REACT_WEB_SEARCH_SYSTEM_PROMPT = """You are a ReAct agent for web search tasks. You use reasoning and acting to find accurate answers to questions.

Available Actions:
1. "search": Execute Google searches for information
2. "verify": Verify if current answer satisfies all criteria in the question
3. "answer": Provide final answer when confident

Your task is to answer questions by iteratively:
- THINKING about what information you need
- ACTING to gather or verify information
- OBSERVING results and updating your understanding

For complex questions with multiple criteria or conditions:
- Start with broad searches to understand the topic
- Refine searches based on observations to get more specific information
- Combine relevant criteria in search queries to narrow results
- Verify your answer addresses all parts of the question
- If verification fails or information is incomplete, refine your search strategy

Response Format:
You must respond with ONLY a JSON object (no other text):

When you need to search for information:
{{
  "thought": "Explain your reasoning about what to search for",
  "action": {{
    "type": "search",
    "queries": ["query 1", "query 2"]
  }}
}}

When you want to verify an answer:
{{
  "thought": "Explain why you want to verify",
  "action": {{
    "type": "verify",
    "answer": "proposed answer",
    "criteria": ["criterion 1", "criterion 2"]
  }}
}}

When you have a confident final answer:
{{
  "thought": "Explain why you're confident",
  "answer": "Your final answer here"
}}

IMPORTANT:
- Always verify multi-criteria answers before providing final answer
- If verification fails, refine your search
- Maximum {max_steps} iterations
"""


class WebSearchReAct(OllamaAgent):
    """
    ReAct-style web search agent.

    Implements the Thought-Action-Observation loop for web search tasks,
    using specialized sub-agents for search, fetch, and synthesis.
    """

    def __init__(
        self,
        orchestrator_model: str = "gemma3:27b",
        worker_model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        max_iterations: int = 5
    ):
        """
        Initialize the ReAct web search agent.

        Args:
            orchestrator_model: Model for reasoning (default: gemma3:27b)
            worker_model: Model for sub-agents (default: gemma3:4b)
            ollama_url: Ollama API URL
            max_iterations: Maximum reasoning iterations (default: 5)
        """
        super().__init__(model=orchestrator_model, ollama_url=ollama_url)

        self._worker_model = worker_model
        self._max_iterations = max_iterations
        self._history: List[str] = []

        # Store ollama_url for sub-agents
        self._ollama_url = ollama_url

        # Sub-agents
        self.search_agent: Optional[SearchAgent] = None
        self.fetch_agent: Optional[FetchAgent] = None
        self.synthesis_agent: Optional[SynthesisAgent] = None
        self.orchestrator: Optional[WebSearchOrchestrator] = None

        self._initialized = False

        # Accumulated results across iterations
        self._all_search_results = []
        self._all_fetched_content = []

    async def initialize(self):
        """Initialize all sub-agents."""
        if self._initialized:
            return

        from .tools import MCPToolWrapper

        # Initialize tool wrapper for MCP servers
        self._tool_wrapper = MCPToolWrapper()
        await self._tool_wrapper.initialize(servers=["google-search", "fetch"])

        # Initialize sub-agents
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

        self.orchestrator = WebSearchOrchestrator(
            model=self.model,
            ollama_url=self._ollama_url
        )

        self._initialized = True
        logger.info("ReAct Web Search Agent initialized")

    async def cleanup(self):
        """Cleanup all resources."""
        if self.search_agent:
            await self.search_agent.cleanup()
        if self.fetch_agent:
            await self.fetch_agent.cleanup()
        if self._tool_wrapper:
            await self._tool_wrapper.cleanup()
        self._initialized = False

    def _build_prompt(self, question: str) -> str:
        """
        Build the ReAct prompt including history.

        Args:
            question: The user's question

        Returns:
            Complete prompt for the LLM
        """
        prompt_parts = [
            REACT_WEB_SEARCH_SYSTEM_PROMPT.format(max_steps=self._max_iterations),
            f"\nQuestion: {question}",
        ]

        if self._history:
            prompt_parts.append("\nPrevious Steps:")
            prompt_parts.append("\n".join(self._history))

        prompt_parts.append("\nWhat is your next step? Respond with JSON only.")

        return "\n".join(prompt_parts)

    def _add_history(self, step_num: int, thought: str = "", action: str = "", observation: str = ""):
        """Add to history."""
        entry_parts = [f"Step {step_num}:"]
        if thought:
            entry_parts.append(f"Thought: {thought}")
        if action:
            entry_parts.append(f"Action: {action}")
        if observation:
            entry_parts.append(f"Observation: {observation}")
        self._history.append("\n".join(entry_parts))

    async def search(
        self,
        question: str,
        output_format: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Execute ReAct-style web search.

        Args:
            question: User's question
            output_format: Expected output format

        Returns:
            Dictionary with 'answer' field
        """
        if not self._initialized:
            await self.initialize()

        logger.info(f"=== ReAct Web Search: {question[:100]}... ===")

        # Reset state
        self._history = []
        self._all_search_results = []
        self._all_fetched_content = []

        for iteration in range(self._max_iterations):
            logger.info(f"\n=== Iteration {iteration + 1}/{self._max_iterations} ===")

            # Build prompt with history
            prompt = self._build_prompt(question)

            # Get LLM decision
            response = self.generate(prompt)
            logger.info(f"LLM Response: {response[:200]}...")

            try:
                # Parse JSON response
                response = response.strip().strip('`').strip()
                if response.startswith("json"):
                    response = response[4:].strip()
                parsed = json.loads(response)

                if "thought" not in parsed:
                    raise ValueError("No 'thought' in response")

                thought = parsed["thought"]
                logger.info(f"Thought: {thought}")

                # Check if LLM provided final answer
                if "answer" in parsed:
                    answer = parsed["answer"]
                    logger.info(f"Final Answer: {answer}")
                    self._add_history(iteration + 1, thought=thought, action="ANSWER", observation=answer)
                    return {"answer": answer}

                # Process action
                if "action" not in parsed:
                    raise ValueError("No 'action' or 'answer' in response")

                action = parsed["action"]
                action_type = action.get("type", "")

                if action_type == "search":
                    # Execute searches
                    queries = action.get("queries", [])
                    logger.info(f"Action: SEARCH with {len(queries)} queries")

                    search_results = await self.search_agent.search(queries, num_results=10)
                    self._all_search_results.extend(search_results)

                    # Fetch top URLs
                    fetched = await self.fetch_agent.fetch_from_search_results(
                        search_results,
                        top_n=5,
                        max_length=10000
                    )
                    self._all_fetched_content.extend(fetched)

                    # Synthesize information
                    extracted = await self.synthesis_agent.synthesize(
                        question=question,
                        search_results=self._all_search_results,
                        fetched_content=self._all_fetched_content,
                        expected_answer_type=None  # Let synthesis agent determine
                    )

                    # Create observation from facts
                    if extracted.facts:
                        facts_preview = "; ".join(extracted.facts[:3])
                        observation = f"Found {len(extracted.facts)} facts: {facts_preview}"
                    else:
                        observation = "No relevant information found in search results"

                    logger.info(f"Observation: {observation}")
                    self._add_history(iteration + 1, thought=thought, action=f"SEARCH: {queries}", observation=observation)

                elif action_type == "verify":
                    # Verify answer against criteria
                    answer = action.get("answer", "")
                    criteria = action.get("criteria", [])
                    logger.info(f"Action: VERIFY answer '{answer}' against {len(criteria)} criteria")

                    # Get current facts
                    if self._all_search_results:
                        extracted = await self.synthesis_agent.synthesize(
                            question=question,
                            search_results=self._all_search_results,
                            fetched_content=self._all_fetched_content,
                            expected_answer_type=None  # Let synthesis agent determine
                        )

                        verification = await self.orchestrator.verify_answer(
                            question=question,
                            answer=answer,
                            facts=extracted.facts
                        )

                        observation = f"Verified: {verification['verified']}, Confidence: {verification['confidence']:.2f}. {verification['reasoning']}"
                        if verification['missing_criteria']:
                            observation += f" Missing: {verification['missing_criteria']}"
                        logger.info(f"Observation: {observation}")

                        self._add_history(iteration + 1, thought=thought, action=f"VERIFY: {answer}", observation=observation)
                    else:
                        observation = "No search results to verify against"
                        self._add_history(iteration + 1, thought=thought, action="VERIFY", observation=observation)

                else:
                    observation = f"Unknown action type: {action_type}"
                    logger.warning(observation)
                    self._add_history(iteration + 1, thought=thought, action=str(action), observation=observation)

            except (json.JSONDecodeError, ValueError, KeyError) as e:
                error_msg = f"Failed to parse LLM response: {e}"
                logger.error(error_msg)
                self._add_history(iteration + 1, thought="", action="ERROR", observation=error_msg)
                continue

        # Max iterations reached
        logger.warning(f"Max iterations ({self._max_iterations}) reached")

        # Try to extract best answer from accumulated facts
        if self._all_search_results:
            extracted = await self.synthesis_agent.synthesize(
                question=question,
                search_results=self._all_search_results,
                fetched_content=self._all_fetched_content,
                expected_answer_type="person_name"
            )
            final_output = await self.synthesis_agent.generate_answer(question, extracted)
            return {"answer": final_output.answer}

        return {"answer": "Unable to find answer within iteration limit"}
