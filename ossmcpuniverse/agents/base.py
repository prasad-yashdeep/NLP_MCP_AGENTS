"""
Base agent class for Ollama-based agents.

Provides a wrapper around Ollama models for use in the multi-agent system,
similar to the OpenAI Agents SDK pattern but adapted for local Ollama models.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, Type, List
from dataclasses import dataclass

import requests
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


@dataclass
class OllamaAgentConfig:
    """Configuration for an Ollama-based agent."""

    name: str = "OllamaAgent"
    model: str = "gemma3:4b"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    temperature: float = 0.3
    max_tokens: int = 2048
    system_prompt: str = "You are a helpful AI assistant."
    timeout: int = 300


class OllamaAgent:
    """
    Base agent class that wraps Ollama models.

    This class provides a similar interface to OpenAI's Agent SDK
    but uses local Ollama models for inference.
    """

    def __init__(
        self,
        name: str = "OllamaAgent",
        model: str = "gemma3:4b",
        system_prompt: str = "You are a helpful AI assistant.",
        temperature: float = 0.3,
        max_tokens: int = 2048,
        ollama_url: Optional[str] = None,
    ):
        """
        Initialize an Ollama agent.

        Args:
            name: Agent name for identification
            model: Ollama model name (e.g., "gemma3:4b", "gemma3:27b")
            system_prompt: System prompt for the agent
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            ollama_url: Ollama API URL (defaults to OLLAMA_URL env var)
        """
        self.name = name
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.logger = logging.getLogger(f"OllamaAgent:{name}")
        self._conversation_history: List[Dict[str, str]] = []

    def reset(self):
        """Reset conversation history."""
        self._conversation_history = []

    def generate(
        self,
        prompt: str,
        response_format: Optional[Type[BaseModel]] = None,
        include_history: bool = False,
        **kwargs
    ) -> Any:
        """
        Generate a response from the Ollama model.

        Args:
            prompt: User prompt
            response_format: Optional Pydantic model for structured output
            include_history: Whether to include conversation history
            **kwargs: Additional arguments

        Returns:
            Generated response (string or Pydantic model instance)
        """
        messages = []

        # Add system prompt
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        # Add conversation history if requested
        if include_history:
            messages.extend(self._conversation_history)

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        # If structured output is requested, add format instruction
        if response_format is not None:
            schema = response_format.model_json_schema()
            properties = schema.get("properties", {})
            format_str = json.dumps({k: v.get("title", k) for k, v in properties.items()})
            messages.append({
                "role": "user",
                "content": f"Respond with ONLY a valid JSON object following this format: {format_str}"
            })

        # Prepare request
        url = f"{self.ollama_url.rstrip('/')}/api/chat"
        data = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }

        # Force JSON format if structured output requested
        if response_format is not None:
            data["format"] = "json"

        try:
            response = requests.post(
                url,
                json=data,
                timeout=kwargs.get("timeout", 300)
            )
            response.raise_for_status()

            json_response = response.json()
            message = json_response.get("message", {})
            content = message.get("content", "")

            # Handle thinking field for reasoning models
            if not content and message.get("thinking"):
                content = message["thinking"]

            # Update conversation history
            self._conversation_history.append({"role": "user", "content": prompt})
            self._conversation_history.append({"role": "assistant", "content": content})

            # Parse response if structured format requested
            if response_format is not None:
                try:
                    parsed = json.loads(content)
                    return response_format.model_validate(parsed)
                except (json.JSONDecodeError, Exception) as e:
                    self.logger.warning(f"Failed to parse structured response: {e}")
                    # Try to extract JSON from the response
                    return self._extract_and_parse(content, response_format)

            return content

        except requests.exceptions.RequestException as e:
            self.logger.error(f"Ollama request failed: {e}")
            raise RuntimeError(f"Failed to generate response: {e}") from e

    def _extract_and_parse(
        self,
        text: str,
        response_format: Type[BaseModel]
    ) -> Optional[BaseModel]:
        """Extract JSON from text and parse into Pydantic model."""
        import re

        # Try to find JSON in code blocks
        patterns = [
            r'```json\s*(.*?)```',
            r'```\s*(\{.*?\})\s*```',
            r'\{[^{}]*\}'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            for match in matches:
                try:
                    parsed = json.loads(match.strip() if isinstance(match, str) else match)
                    return response_format.model_validate(parsed)
                except (json.JSONDecodeError, Exception):
                    continue

        # Try parsing the whole text as JSON
        try:
            # Find content between first { and last }
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1:
                json_str = text[start:end+1]
                parsed = json.loads(json_str)
                return response_format.model_validate(parsed)
        except (json.JSONDecodeError, Exception):
            pass

        self.logger.error(f"Could not extract valid JSON from response: {text[:200]}...")
        return None

    async def generate_async(
        self,
        prompt: str,
        response_format: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Any:
        """
        Async version of generate (currently wraps sync version).

        For true async support, would need aiohttp.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.generate(prompt, response_format, **kwargs)
        )

    def __repr__(self) -> str:
        return f"OllamaAgent(name='{self.name}', model='{self.model}')"
