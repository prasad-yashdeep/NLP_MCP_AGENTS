"""
Orchestrator Agent - the brain of the multi-agent system.

This agent uses gemma3:27b to parse user queries, decompose tasks,
coordinate sub-agents, and synthesize final answers.
"""

import logging
import json
import re
from typing import Optional, Dict, Any, List

from .base import OllamaAgent
from .models import TaskPlan

logger = logging.getLogger(__name__)


ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent - the brain of a multi-agent financial analysis system.

Your responsibilities:
1. Parse user questions about stock investments
2. Extract key information: tickers, dates, investment amounts, allocation splits
3. Create a structured task plan for sub-agents

When analyzing a question, identify:
- Stock ticker symbol(s) (e.g., MSFT, AAPL, GOOGL)
- Start date and end date (in YYYY-MM-DD format)
- Investment amount (in dollars)
- Allocation splits (how investment is divided across stocks)

For single-stock investments, the split is [1.0].
For multiple stocks, splits should sum to 1.0 (e.g., [0.5, 0.5] for equal split).

Always respond with a JSON object containing the extracted information.
"""


class OrchestratorAgent(OllamaAgent):
    """
    Orchestrator Agent that coordinates the multi-agent system.

    Uses gemma3:27b (the larger model) for complex reasoning including:
    - Query understanding and task decomposition
    - Coordinating sub-agents
    - Synthesizing final answers
    """

    def __init__(
        self,
        model: str = "gemma3:27b",
        ollama_url: Optional[str] = None
    ):
        """
        Initialize the Orchestrator Agent.

        Args:
            model: Ollama model to use (default: gemma3:27b)
            ollama_url: Ollama API URL
        """
        super().__init__(
            name="OrchestratorAgent",
            model=model,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT,
            temperature=0.3,  # Moderate temperature for reasoning
            max_tokens=2048,
            ollama_url=ollama_url
        )

    def plan(self, query: str, output_format: Optional[Dict] = None) -> TaskPlan:
        """
        Parse a user query and create a task plan.

        Args:
            query: User's financial analysis question
            output_format: Expected output format from the benchmark

        Returns:
            TaskPlan with extracted information
        """
        # First try to extract information using regex patterns
        plan = self._extract_with_patterns(query)

        if plan:
            if output_format:
                plan.output_format = output_format
            logger.info(f"Extracted plan using patterns: {plan}")
            return plan

        # Fall back to LLM-based extraction
        plan = self._extract_with_llm(query, output_format)
        logger.info(f"Extracted plan using LLM: {plan}")
        return plan

    def _extract_with_patterns(self, query: str) -> Optional[TaskPlan]:
        """
        Extract task information using regex patterns.

        This is faster and more reliable for well-structured queries.
        """
        try:
            # Extract investment amount
            amount_patterns = [
                r'\$?([\d,]+(?:\.\d{2})?)\s*(?:investment|invested|in)',
                r'(?:invest|invested|investment\s+of)\s*\$?([\d,]+(?:\.\d{2})?)',
                r'\$?([\d,]+(?:\.\d{2})?)\s*(?:worth|total)',
            ]
            investment = None
            for pattern in amount_patterns:
                match = re.search(pattern, query, re.IGNORECASE)
                if match:
                    investment = float(match.group(1).replace(',', ''))
                    break

            if investment is None:
                return None

            # Extract ticker symbols
            ticker_pattern = r'\b([A-Z]{1,5})\b'
            potential_tickers = re.findall(ticker_pattern, query)
            # Filter common words that aren't tickers
            non_tickers = {
                'IF', 'IN', 'OF', 'TO', 'ON', 'AT', 'BY', 'FROM', 'AND', 'THE', 'FOR',
                'WITH', 'AS', 'OR', 'AN', 'A', 'I', 'IS', 'IT', 'BE', 'MY', 'WAS',
                'WHAT', 'WHEN', 'HOW', 'WHICH', 'THAT', 'THIS', 'THEN', 'THAN', 'SO',
                'UP', 'DOWN', 'ALL', 'EACH', 'SOME', 'ANY', 'NO', 'NOT', 'ONLY',
                'INTO', 'OVER', 'UNDER', 'AFTER', 'BEFORE', 'SINCE', 'UNTIL', 'WHILE',
                'ABOUT', 'ABOVE', 'BELOW', 'BETWEEN', 'THROUGH', 'DURING', 'TOWARD',
                'USD', 'ETF', 'IPO', 'CEO', 'CFO', 'CTO', 'COO', 'ROI', 'YTD', 'QTD'
            }
            tickers = [t for t in potential_tickers if t not in non_tickers]

            if not tickers:
                return None

            # Extract dates (YYYY-MM-DD format)
            date_pattern = r'(\d{4}-\d{2}-\d{2})'
            dates = re.findall(date_pattern, query)

            if len(dates) < 2:
                # Try alternative date formats
                alt_date_pattern = r'(\d{1,2}/\d{1,2}/\d{4})'
                alt_dates = re.findall(alt_date_pattern, query)
                if alt_dates:
                    # Convert MM/DD/YYYY to YYYY-MM-DD
                    for d in alt_dates:
                        parts = d.split('/')
                        if len(parts) == 3:
                            dates.append(f"{parts[2]}-{parts[0]:0>2}-{parts[1]:0>2}")

            if len(dates) < 2:
                return None

            start_date = min(dates)
            end_date = max(dates)

            # Determine splits
            # Look for percentage patterns
            split_pattern = r'(\d+(?:\.\d+)?)\s*%'
            percentages = re.findall(split_pattern, query)

            if percentages and len(percentages) == len(tickers):
                splits = [float(p) / 100.0 for p in percentages]
            elif 'equally' in query.lower() or 'equal' in query.lower():
                splits = [1.0 / len(tickers)] * len(tickers)
            elif len(tickers) == 1:
                splits = [1.0]
            else:
                # Default to equal splits
                splits = [1.0 / len(tickers)] * len(tickers)

            return TaskPlan(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date,
                investment=investment,
                splits=splits
            )

        except Exception as e:
            logger.warning(f"Pattern extraction failed: {e}")
            return None

    def _extract_with_llm(
        self,
        query: str,
        output_format: Optional[Dict] = None
    ) -> TaskPlan:
        """
        Extract task information using the LLM.

        Used as a fallback when pattern matching fails.
        """
        prompt = f"""Analyze this financial question and extract the key information:

Question: {query}

Extract:
1. Stock ticker symbol(s)
2. Start date (YYYY-MM-DD format)
3. End date (YYYY-MM-DD format)
4. Investment amount (number only, no $ sign)
5. Allocation splits (list of decimals that sum to 1.0)

Respond with ONLY a JSON object:
{{
    "tickers": ["TICKER1", "TICKER2"],
    "start_date": "YYYY-MM-DD",
    "end_date": "YYYY-MM-DD",
    "investment": 00000,
    "splits": [0.5, 0.5]
}}"""

        response = self.generate(prompt, response_format=TaskPlan)

        if isinstance(response, TaskPlan):
            if output_format:
                response.output_format = output_format
            return response

        # If we got a string, try to parse it
        if isinstance(response, str):
            try:
                # Find JSON in response
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                    plan = TaskPlan(
                        tickers=data.get("tickers", []),
                        start_date=data.get("start_date", ""),
                        end_date=data.get("end_date", ""),
                        investment=float(data.get("investment", 0)),
                        splits=data.get("splits", [1.0]),
                        output_format=output_format
                    )
                    return plan
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse LLM response: {e}")

        # Return a default plan if all extraction fails
        logger.error("All extraction methods failed, returning default plan")
        return TaskPlan(
            tickers=["UNKNOWN"],
            start_date="2024-01-01",
            end_date="2024-12-31",
            investment=0,
            splits=[1.0],
            output_format=output_format
        )

    async def plan_async(
        self,
        query: str,
        output_format: Optional[Dict] = None
    ) -> TaskPlan:
        """Async version of plan."""
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.plan(query, output_format)
        )

    def synthesize(
        self,
        query: str,
        plan: TaskPlan,
        result: Dict[str, Any]
    ) -> str:
        """
        Synthesize a human-readable summary of the analysis.

        Args:
            query: Original user question
            plan: Task plan that was executed
            result: Final calculation result

        Returns:
            Human-readable summary
        """
        prompt = f"""Synthesize a brief answer for this financial analysis:

Original Question: {query}

Analysis Details:
- Stocks: {', '.join(plan.tickers)}
- Period: {plan.start_date} to {plan.end_date}
- Investment: ${plan.investment:,.2f}
- Allocation: {', '.join(f'{s*100:.1f}%' for s in plan.splits)}

Results:
- Final Value: ${result.get('total_value', 0):,.2f}
- Return: {result.get('percentage_return', 0):.2f}%

Write a 1-2 sentence summary of the investment outcome."""

        return self.generate(prompt)

    def __repr__(self) -> str:
        return f"OrchestratorAgent(model='{self.model}')"
