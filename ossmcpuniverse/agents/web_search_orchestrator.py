"""
Web Search Orchestrator - the brain of the web search multi-agent system.

This agent uses gemma3:27b to parse user queries, decompose complex questions,
coordinate sub-agents, and plan search strategies.
"""

import logging
import json
import re
from typing import Optional, List

from .base import OllamaAgent
from .models import SearchPlan

logger = logging.getLogger(__name__)


WEB_SEARCH_ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent - the brain of a multi-agent web search system.

Your responsibilities:
1. Analyze user information retrieval questions
2. Decompose complex multi-hop queries into simpler search queries
3. Create a structured search plan for sub-agents
4. Identify the type of answer expected (person name, date, number, etc.)

When analyzing a question, identify:
- The main question to answer
- Key entities, relationships, and constraints mentioned
- Sub-questions that need to be answered to solve the main question
- The expected type of answer (person_name, organization, date, number, etc.)

For multi-hop questions (requiring multiple searches):
- Break down into logical search steps
- Each search should build on previous results
- Order searches from broad to specific

Always respond with a JSON object containing the search plan.
"""


class WebSearchOrchestrator(OllamaAgent):
    """
    Orchestrator Agent for web search tasks.

    Uses gemma3:27b (the larger model) for complex reasoning including:
    - Query understanding and decomposition
    - Search strategy planning
    - Coordinating sub-agents
    """

    def __init__(
        self,
        model: str = "gemma3:27b",
        ollama_url: Optional[str] = None
    ):
        """
        Initialize the Web Search Orchestrator.

        Args:
            model: Ollama model to use (default: gemma3:27b)
            ollama_url: Ollama API URL
        """
        super().__init__(
            name="WebSearchOrchestrator",
            model=model,
            system_prompt=WEB_SEARCH_ORCHESTRATOR_SYSTEM_PROMPT,
            temperature=0.3,  # Moderate temperature for reasoning
            max_tokens=2048,
            ollama_url=ollama_url
        )

    def plan(self, query: str) -> SearchPlan:
        """
        Parse a user query and create a search plan.

        Args:
            query: User's information retrieval question

        Returns:
            SearchPlan with decomposed queries and strategy
        """
        # Use LLM to decompose the query
        prompt = self._build_planning_prompt(query)
        response = self.generate(prompt)

        # Parse the response
        search_plan = self._parse_planning_response(response, query)

        logger.info(f"Search Plan: {search_plan.search_queries}")
        return search_plan

    def _build_planning_prompt(self, query: str) -> str:
        """Build a prompt for query decomposition."""
        prompt = f"""Analyze this information retrieval question and create a search plan:

Question: {query}

Tasks:
1. Identify what type of answer is needed (person_name, organization, date, number, location, etc.)
2. Break down the question into 2-4 specific search queries
3. Order the queries from broad to specific or in logical search order
4. Describe the overall search strategy

Respond ONLY with this JSON format:
{{
  "main_query": "The original question",
  "expected_answer_type": "person_name",
  "search_queries": [
    "first search query",
    "second search query",
    "third search query"
  ],
  "strategy": "Brief description of search approach"
}}

Examples:

Question: Who is the current CEO of Microsoft?
{{
  "main_query": "Who is the current CEO of Microsoft?",
  "expected_answer_type": "person_name",
  "search_queries": [
    "Microsoft CEO current 2024",
    "Satya Nadella Microsoft CEO"
  ],
  "strategy": "Search for current Microsoft CEO and verify with recent information"
}}

Question: Find the person with 16 goals in 2024-25, 1 UCL goal, 11 goals in 2021-22, 2 EFL Cup goals in 2020-21
{{
  "main_query": "Find the person with 16 goals in 2024-25, 1 UCL goal, 11 goals in 2021-22, 2 EFL Cup goals in 2020-21",
  "expected_answer_type": "person_name",
  "search_queries": [
    "football player 16 goals 2024-25 season",
    "player 1 UCL goal 2024-25 11 goals 2021-22",
    "footballer 2 EFL Cup goals 2020-21 statistics"
  ],
  "strategy": "Search for footballers matching each season's statistics, cross-reference to find player matching all criteria"
}}

Now analyze this question:
{query}

Provide ONLY the JSON, nothing else."""

        return prompt

    def _parse_planning_response(
        self,
        response: str,
        original_query: str
    ) -> SearchPlan:
        """Parse the LLM response and extract search plan."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                search_queries = data.get("search_queries", [])
                if isinstance(search_queries, str):
                    search_queries = [search_queries]

                return SearchPlan(
                    main_query=data.get("main_query", original_query),
                    search_queries=search_queries,
                    strategy=data.get("strategy", "Standard web search"),
                    expected_answer_type=data.get("expected_answer_type")
                )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse planning response: {e}")

        # Fallback: use original query as single search
        logger.info("Using fallback search plan with original query")
        return SearchPlan(
            main_query=original_query,
            search_queries=[original_query],
            strategy="Single search query",
            expected_answer_type=self._infer_answer_type(original_query)
        )

    def _infer_answer_type(self, query: str) -> str:
        """Infer the expected answer type from the query."""
        query_lower = query.lower()

        if query_lower.startswith("who"):
            return "person_name"
        elif query_lower.startswith("when") or "date" in query_lower:
            return "date"
        elif query_lower.startswith("where") or "location" in query_lower:
            return "location"
        elif query_lower.startswith("how many") or "number" in query_lower:
            return "number"
        elif "company" in query_lower or "organization" in query_lower:
            return "organization"
        else:
            return "unknown"

    async def plan_async(self, query: str) -> SearchPlan:
        """Async version of plan."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.plan(query)
        )

    def refine_plan(
        self,
        initial_plan: SearchPlan,
        intermediate_results: List[str]
    ) -> SearchPlan:
        """
        Refine the search plan based on intermediate results.

        This allows for iterative multi-hop search where later queries
        can be informed by earlier search results.

        Args:
            initial_plan: The original search plan
            intermediate_results: Summary of results from earlier searches

        Returns:
            Refined SearchPlan with additional queries
        """
        prompt = f"""Given the initial search plan and intermediate results, refine the search strategy:

Original Question: {initial_plan.main_query}
Initial Search Queries: {initial_plan.search_queries}

Intermediate Results:
{chr(10).join(f"- {result}" for result in intermediate_results)}

Based on these results, should we:
1. Continue with the existing plan?
2. Add more specific search queries?
3. Change the search direction?

Respond with a JSON object containing additional search queries if needed:
{{
  "additional_queries": ["query 1", "query 2"],
  "reasoning": "Why these queries are needed"
}}

If no additional queries are needed, return:
{{
  "additional_queries": [],
  "reasoning": "Sufficient information obtained"
}}"""

        response = self.generate(prompt)

        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                additional = data.get("additional_queries", [])

                if additional:
                    # Add to existing plan
                    initial_plan.search_queries.extend(additional)
                    logger.info(f"Refined plan with {len(additional)} additional queries")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse refinement response: {e}")

        return initial_plan

    async def verify_answer(
        self,
        question: str,
        answer: str,
        facts: List[str]
    ) -> dict:
        """
        Verify if the answer is correct and complete based on evidence.

        For multi-criteria questions, this checks if the answer satisfies ALL criteria.

        Args:
            question: The original question
            answer: The proposed answer
            facts: Supporting facts/evidence

        Returns:
            {
                "verified": bool - whether answer is verified correct
                "confidence": float - confidence in verification (0.0-1.0)
                "reasoning": str - explanation of verification
                "should_refine": bool - whether search should be refined
                "missing_criteria": List[str] - criteria not verified
            }
        """
        prompt = f"""You are verifying if an answer is correct and complete.

Question: {question}

Proposed Answer: {answer}

Supporting Facts:
{chr(10).join(f"- {fact}" for fact in facts)}

Your task:
1. Check if the answer directly answers the question
2. For multi-criteria questions, verify the answer satisfies ALL criteria mentioned
3. Assess if the facts provide sufficient evidence

Respond with JSON:
{{
  "verified": true or false,
  "confidence": 0.0 to 1.0,
  "reasoning": "Explanation of verification",
  "should_refine": true or false,
  "missing_criteria": ["criterion 1", "criterion 2"]
}}

Important:
- If answer is "Unknown", "No information", or similar: verified=false, should_refine=true
- If facts contradict some criteria: verified=false, should_refine=true
- If facts support ALL criteria: verified=true, should_refine=false
- List any criteria that couldn't be verified in missing_criteria

Provide ONLY the JSON."""

        response = self.generate(prompt)

        try:
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                return {
                    "verified": data.get("verified", False),
                    "confidence": float(data.get("confidence", 0.0)),
                    "reasoning": data.get("reasoning", ""),
                    "should_refine": data.get("should_refine", True),
                    "missing_criteria": data.get("missing_criteria", [])
                }
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse verification response: {e}")

        # Default: not verified
        return {
            "verified": False,
            "confidence": 0.0,
            "reasoning": "Failed to verify answer",
            "should_refine": True,
            "missing_criteria": []
        }

    def __repr__(self) -> str:
        return f"WebSearchOrchestrator(model='{self.model}')"
