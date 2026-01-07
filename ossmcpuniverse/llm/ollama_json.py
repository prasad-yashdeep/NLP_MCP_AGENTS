"""
Custom Ollama LLM wrapper that forces JSON output format for reasoning models.
This wrapper extracts JSON from model responses that may contain chain-of-thought reasoning.
"""
import os
import sys
import json
import re
import logging
from dataclasses import dataclass
from typing import Dict, Union, Optional, Type, List

import requests
from dotenv import load_dotenv
from pydantic import BaseModel as PydanticBaseModel

# Add MCP-Universe to path
MCP_UNIVERSE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + "/MCP-Universe"
sys.path.insert(0, MCP_UNIVERSE_PATH)

from mcpuniverse.common.config import BaseConfig
from mcpuniverse.common.context import Context
from mcpuniverse.llm.base import BaseLLM

load_dotenv()


def extract_json_from_text(text: str) -> Optional[str]:
    """
    Extract JSON object or array from text that may contain other content.
    Handles reasoning model outputs that wrap JSON in chain-of-thought.
    """
    if not text:
        return None

    # Try to find JSON object or array
    # First, try to parse the entire text as JSON
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Look for JSON patterns in the text
    # Try to find the first complete JSON object {...}
    brace_count = 0
    start_idx = -1

    for i, char in enumerate(text):
        if char == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx >= 0:
                potential_json = text[start_idx:i+1]
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    # Continue looking for other JSON objects
                    start_idx = -1

    return None


@dataclass
class OllamaJSONConfig(BaseConfig):
    """
    Configuration for Ollama JSON models (for reasoning models like deepseek-r1, gpt-oss).
    """
    model_name: str = "deepseek-r1:14b"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    temperature: float = 0.6
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    max_completion_tokens: int = 4096
    seed: int = 12345
    reasoning: str = "high"  # For compatibility with HarmonyReAct


class OllamaJSONModel(BaseLLM):
    """
    Ollama LLM wrapper that forces JSON output format.
    Designed for reasoning models like deepseek-r1 and gpt-oss that may output
    chain-of-thought reasoning before their JSON response.
    """
    config_class = OllamaJSONConfig
    alias = "ollama_json"
    env_vars = ["OLLAMA_URL"]

    def __init__(self, config: Optional[Union[Dict, str]] = None):
        super().__init__()
        self.config = OllamaJSONModel.config_class.load(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def _generate(
            self,
            messages: List[dict[str, str]],
            response_format: Type[PydanticBaseModel] = None,
            **kwargs
    ):
        """
        Generate content using Ollama with forced JSON format.
        Extracts JSON from response even if model outputs chain-of-thought.
        """
        ollama_url = self.config.ollama_url
        url = ollama_url.strip("/") + "/api/chat"

        # Force JSON format in the request
        data = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "format": "json",  # Force JSON output
            "options": {
                "seed": self.config.seed,
                "num_predict": self.config.max_completion_tokens,
                "top_p": self.config.top_p,
                "temperature": self.config.temperature,
                "presence_penalty": self.config.presence_penalty,
                "frequency_penalty": self.config.frequency_penalty,
            }
        }

        try:
            response = requests.post(url, json=data, timeout=int(kwargs.get("timeout", 300)))
            json_data = json.loads(response.text)
            message = json_data.get("message", {})

            # Get content from either content or thinking field
            content = message.get("content", "")
            thinking = message.get("thinking", "")

            # Prefer content if available, otherwise use thinking
            if content:
                result = content
            elif thinking:
                result = thinking
            else:
                self.logger.warning("No content or thinking in response")
                return None

            # Try to extract JSON if the response isn't already valid JSON
            extracted = extract_json_from_text(result)
            if extracted:
                return extracted

            # If extraction failed, return the raw content
            # The ReAct agent will handle the parsing error
            return result

        except Exception as e:
            self.logger.error("Error in generation: %s", str(e))
            return None

    def set_context(self, context: Context):
        """Set context, e.g., environment variables (API keys)."""
        super().set_context(context)
        self.config.ollama_url = context.env.get("OLLAMA_URL", self.config.ollama_url)
