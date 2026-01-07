"""
Synthesis Agent for extracting and synthesizing information.

This specialized agent uses gemma3:4b to analyze search results and fetched
content, extract relevant facts, and generate final answers.
"""

import logging
import json
import re
from typing import List, Optional

from .base import OllamaAgent
from .models import SearchResult, FetchedContent, ExtractedInfo, WebSearchOutput

logger = logging.getLogger(__name__)


SYNTHESIS_AGENT_SYSTEM_PROMPT = """You are a specialized Synthesis Agent responsible for analyzing information and generating answers.

Your task is to:
1. Analyze search results and fetched webpage content
2. Extract relevant facts that answer the user's question
3. Cross-reference information across multiple sources
4. Generate a concise, accurate answer
5. Assess confidence in the answer based on source agreement

Always prioritize accuracy and cite information from reliable sources.
Focus on extracting the most relevant information to answer the specific question.
"""


class SynthesisAgent(OllamaAgent):
    """
    Specialized agent for synthesizing information and generating answers.

    Uses gemma3:4b to analyze search results, extract facts, and generate
    final answers based on retrieved information.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None
    ):
        """
        Initialize the Synthesis Agent.

        Args:
            model: Ollama model to use (default: gemma3:4b)
            ollama_url: Ollama API URL
        """
        super().__init__(
            name="SynthesisAgent",
            model=model,
            system_prompt=SYNTHESIS_AGENT_SYSTEM_PROMPT,
            temperature=0.3,  # Moderate temperature for synthesis
            max_tokens=2048,
            ollama_url=ollama_url
        )

    async def synthesize(
        self,
        question: str,
        search_results: List[SearchResult],
        fetched_content: List[FetchedContent],
        expected_answer_type: Optional[str] = None
    ) -> ExtractedInfo:
        """
        Synthesize information from search results and fetched content.

        Args:
            question: The original question to answer
            search_results: Search results from SearchAgent
            fetched_content: Fetched webpage content from FetchAgent
            expected_answer_type: Type of answer expected (e.g., 'person_name')

        Returns:
            ExtractedInfo with facts, sources, and confidence
        """
        # Build a comprehensive prompt with all available information
        prompt = self._build_synthesis_prompt(
            question,
            search_results,
            fetched_content,
            expected_answer_type
        )

        # Generate analysis using LLM
        response = self.generate(prompt)

        # Parse the response to extract facts
        extracted_info = self._parse_synthesis_response(
            response,
            search_results,
            fetched_content
        )

        logger.info(
            f"Synthesized {len(extracted_info.facts)} facts "
            f"from {len(fetched_content)} sources "
            f"(confidence: {extracted_info.confidence:.2f})"
        )

        return extracted_info

    def _build_synthesis_prompt(
        self,
        question: str,
        search_results: List[SearchResult],
        fetched_content: List[FetchedContent],
        expected_answer_type: Optional[str] = None
    ) -> str:
        """Build a comprehensive prompt for synthesis."""
        prompt_parts = [
            f"Question: {question}",
            ""
        ]

        if expected_answer_type:
            prompt_parts.append(f"Expected answer type: {expected_answer_type}")
            prompt_parts.append("")

        # Add search result snippets
        if search_results:
            prompt_parts.append("Search Results:")
            for i, sr in enumerate(search_results[:3], 1):  # Top 3 searches
                prompt_parts.append(f"\nSearch Query {i}: {sr.query}")
                for j, result in enumerate(sr.results[:5], 1):  # Top 5 results
                    prompt_parts.append(
                        f"  {j}. {result.get('title', 'No title')}"
                    )
                    snippet = result.get('snippet', '')
                    if snippet:
                        prompt_parts.append(f"     {snippet[:200]}")
            prompt_parts.append("")

        # Add fetched content
        if fetched_content:
            prompt_parts.append("Fetched Content:")
            for i, content in enumerate(fetched_content[:5], 1):  # Top 5 pages
                if content.status == "success" and content.content:
                    prompt_parts.append(f"\n{i}. {content.title or content.url}")
                    # Truncate content to first 500 chars
                    content_preview = content.content[:500].replace('\n', ' ')
                    prompt_parts.append(f"   {content_preview}...")
            prompt_parts.append("")

        # Add instructions
        prompt_parts.extend([
            "Based on the search results and fetched content above:",
            "1. Extract relevant facts that answer the question",
            "2. Identify the most likely answer",
            "3. List the sources (URLs) used",
            "4. Assess your confidence (0.0 to 1.0)",
            "",
            "Respond in this JSON format:",
            "{",
            '  "answer": "Your concise answer here",',
            '  "facts": ["fact 1", "fact 2", "fact 3"],',
            '  "confidence": 0.85,',
            '  "reasoning": "Brief explanation of how you arrived at this answer"',
            "}",
            "",
            "Provide ONLY the JSON, no additional text."
        ])

        return "\n".join(prompt_parts)

    def _parse_synthesis_response(
        self,
        response: str,
        search_results: List[SearchResult],
        fetched_content: List[FetchedContent]
    ) -> ExtractedInfo:
        """Parse the LLM response and extract structured information."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())

                facts = data.get("facts", [])
                if isinstance(facts, str):
                    facts = [facts]

                # Collect source URLs
                sources = []
                for content in fetched_content:
                    if content.status == "success":
                        sources.append(content.url)

                logger.info(f"Extracted {len(facts)} facts from synthesis")
                for i, fact in enumerate(facts, 1):
                    logger.info(f"  Fact {i}: {fact}")

                return ExtractedInfo(
                    facts=facts,
                    sources=sources[:5],  # Top 5 sources
                    confidence=float(data.get("confidence", 0.5)),
                    reasoning=data.get("reasoning", "")
                )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.warning(f"Failed to parse synthesis response as JSON: {e}")

        # Fallback: treat response as answer text
        sources = [c.url for c in fetched_content if c.status == "success"]
        return ExtractedInfo(
            facts=[response[:200]],  # Use first 200 chars as fact
            sources=sources[:5],
            confidence=0.3,  # Low confidence for unparsed response
            reasoning="Failed to parse structured response"
        )

    async def generate_answer(
        self,
        question: str,
        extracted_info: ExtractedInfo
    ) -> WebSearchOutput:
        """
        Generate the final answer from extracted information.

        Args:
            question: The original question
            extracted_info: Information extracted from sources

        Returns:
            WebSearchOutput with final answer
        """
        logger.info(f"Generating answer from {len(extracted_info.facts)} facts")

        if not extracted_info.facts:
            logger.warning("No facts available to generate answer")
            return WebSearchOutput(
                answer="No information found",
                confidence=0.0,
                sources=[]
            )

        # Determine the answer type from the question
        answer_type = self._determine_answer_type(question)
        logger.info(f"Question type detected: {answer_type}")

        # Use LLM to extract the specific answer from facts
        prompt = f"""Based on the extracted facts below, answer this question with ONLY the direct answer.

Question: {question}
Expected Answer Type: {answer_type}

Facts:
{chr(10).join(f"- {fact}" for fact in extracted_info.facts)}

IMPORTANT INSTRUCTIONS:
- For "who" questions about a person: return ONLY the person's full name (e.g., "John Smith")
- For "when" questions: return ONLY the date or time period (e.g., "2024-01-15")
- For "where" questions: return ONLY the location (e.g., "New York City")
- For "what" questions: return ONLY the specific thing asked (e.g., "Microsoft")
- For "how many/much" questions: return ONLY the number (e.g., "42")
- Do NOT include explanations, context, sources, or phrases like "According to..."
- Do NOT include full sentences - just the answer
- If the answer is a person's name, return ONLY the name without any titles or additional info

Answer:"""

        logger.debug(f"Answer generation prompt:\n{prompt}")
        answer = self.generate(prompt)
        logger.info(f"Raw answer from LLM: '{answer}'")

        # Clean up the answer - remove any leading/trailing quotes, periods, etc.
        answer = answer.strip().strip('"').strip("'").strip('.')
        logger.info(f"Final cleaned answer: '{answer}'")

        return WebSearchOutput(
            answer=answer,
            confidence=extracted_info.confidence,
            sources=extracted_info.sources
        )

    def _determine_answer_type(self, question: str) -> str:
        """Determine the type of answer expected from the question."""
        q_lower = question.lower()

        if q_lower.startswith("who") or "looking for someone" in q_lower:
            return "person name (full name only, no titles or context)"
        elif q_lower.startswith("when") or "date" in q_lower:
            return "date or time period"
        elif q_lower.startswith("where") or "location" in q_lower:
            return "location or place name"
        elif q_lower.startswith("what"):
            return "specific thing, entity, or concept"
        elif q_lower.startswith("how many") or q_lower.startswith("how much"):
            return "number or quantity"
        elif q_lower.startswith("which"):
            return "specific choice or selection"
        else:
            return "concise, specific answer"

    def __repr__(self) -> str:
        return f"SynthesisAgent(model='{self.model}')"
