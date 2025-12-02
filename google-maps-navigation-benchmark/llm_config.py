"""
LLM Configuration for CrewAI with Ollama.
Configures CrewAI to use local Gemma 2 models via Ollama.

NOTE: We use OPENAI_API_KEY and OPENAI_API_BASE because:
- CrewAI 1.6.1 uses litellm internally for LLM calls
- litellm uses OpenAI-compatible API format
- Ollama provides an OpenAI-compatible endpoint at /v1
- The "ollama" value for OPENAI_API_KEY is just a placeholder (Ollama doesn't require it)
- This is NOT calling OpenAI - it's using Ollama's OpenAI-compatible interface
"""
import os
from config import CREWAI_MODEL, OLLAMA_BASE_URL, CREWAI_TEMPERATURE

def configure_crewai_for_ollama():
    """
    Configure CrewAI environment variables for Ollama.
    Call this before creating agents.
    CrewAI 1.6.1 uses litellm internally, which supports Ollama via OpenAI-compatible API.
    
    IMPORTANT: This is NOT using OpenAI. We're using Ollama's OpenAI-compatible endpoint.
    """
    # Set up for litellm (which CrewAI uses internally)
    # Format: ollama/gemma2:9b or ollama/gemma2:2b
    model_name = f"ollama/{CREWAI_MODEL}"
    
    # Configure environment variables for litellm to use Ollama
    # Ollama provides OpenAI-compatible API at /v1 endpoint
    os.environ["OPENAI_API_BASE"] = f"{OLLAMA_BASE_URL}/v1"
    # This is a placeholder - Ollama doesn't actually check this key
    # But litellm requires it to be set
    os.environ["OPENAI_API_KEY"] = "ollama"
    os.environ["LITELLM_MODEL"] = model_name
    
    # Also set for CrewAI's direct use
    os.environ["MODEL_NAME"] = model_name
    
    print(f"[INFO] Configured to use Ollama (not OpenAI)")
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Ollama URL: {OLLAMA_BASE_URL}")
    
    return model_name

