"""
Formatter Agent for output formatting.

This specialized agent uses gemma3:4b to format calculation results
into the expected benchmark output format.
"""

import logging
import json
from typing import Optional, Dict, Any

from .base import OllamaAgent
from .models import CalculationResult, FinalOutput

logger = logging.getLogger(__name__)


FORMATTER_AGENT_SYSTEM_PROMPT = """You are a specialized Formatter Agent for financial output.

Your task is to:
1. Format numerical results with appropriate precision
2. Ensure output matches the expected JSON structure
3. Validate that all required fields are present

Output format must be:
{
    "total value": "<number with 2 decimal places>",
    "total percentage return": "<number with 2 decimal places>"
}

Rules:
- Round total value to 2 decimal places
- Round percentage return to 2 decimal places
- Do not include currency symbols or percent signs
- Output only valid JSON
"""


class FormatterAgent(OllamaAgent):
    """
    Specialized agent for formatting financial output.

    Uses gemma3:4b to ensure output matches the expected
    benchmark format with proper formatting and validation.
    """

    def __init__(
        self,
        model: str = "gemma3:4b",
        ollama_url: Optional[str] = None,
        use_llm_formatting: bool = False
    ):
        """
        Initialize the Formatter Agent.

        Args:
            model: Ollama model to use (default: gemma3:4b)
            ollama_url: Ollama API URL
            use_llm_formatting: Whether to use LLM for formatting (vs pure Python)
        """
        super().__init__(
            name="FormatterAgent",
            model=model,
            system_prompt=FORMATTER_AGENT_SYSTEM_PROMPT,
            temperature=0.1,  # Very low temperature for formatting
            max_tokens=512,
            ollama_url=ollama_url
        )
        self._use_llm_formatting = use_llm_formatting

    async def format(
        self,
        calculation: CalculationResult,
        output_format: Optional[Dict[str, str]] = None
    ) -> FinalOutput:
        """
        Format calculation results into final output.

        Args:
            calculation: CalculationResult from Calculator Agent
            output_format: Optional custom output format specification

        Returns:
            FinalOutput with formatted values
        """
        if self._use_llm_formatting:
            return await self._format_with_llm(calculation, output_format)
        else:
            return self._format_direct(calculation, output_format)

    def _format_direct(
        self,
        calculation: CalculationResult,
        output_format: Optional[Dict[str, str]] = None
    ) -> FinalOutput:
        """
        Format results directly using Python.

        This is faster and more reliable than LLM formatting.
        """
        # Format with 2 decimal places
        total_value = f"{calculation.total_value:.2f}"
        percentage_return = f"{calculation.percentage_return:.2f}"

        logger.info(
            f"Formatted output: value=${total_value}, return={percentage_return}%"
        )

        return FinalOutput(
            total_value=total_value,
            total_percentage_return=percentage_return
        )

    async def _format_with_llm(
        self,
        calculation: CalculationResult,
        output_format: Optional[Dict[str, str]] = None
    ) -> FinalOutput:
        """
        Format results using LLM for complex formatting needs.
        """
        prompt = f"""Format these calculation results:

Total Value: {calculation.total_value}
Percentage Return: {calculation.percentage_return}

{"Expected format: " + json.dumps(output_format) if output_format else ""}

Output the formatted JSON:"""

        try:
            response = self.generate(prompt, response_format=FinalOutput)

            if isinstance(response, FinalOutput):
                return response

            # If LLM formatting fails, fall back to direct formatting
            logger.warning("LLM formatting returned invalid type, using direct formatting")
            return self._format_direct(calculation, output_format)

        except Exception as e:
            logger.error(f"LLM formatting failed: {e}")
            return self._format_direct(calculation, output_format)

    def validate_output(self, output: FinalOutput) -> bool:
        """
        Validate the formatted output.

        Args:
            output: FinalOutput to validate

        Returns:
            True if valid, False otherwise
        """
        try:
            # Check total value is a valid number
            float(output.total_value)

            # Check percentage return is a valid number
            float(output.total_percentage_return)

            return True

        except (ValueError, TypeError):
            return False

    def to_benchmark_format(self, output: FinalOutput) -> Dict[str, str]:
        """
        Convert FinalOutput to the format expected by the benchmark.

        Args:
            output: FinalOutput object

        Returns:
            Dictionary matching benchmark expected format
        """
        return output.to_benchmark_format()

    def __repr__(self) -> str:
        return f"FormatterAgent(model='{self.model}', llm_formatting={self._use_llm_formatting})"
