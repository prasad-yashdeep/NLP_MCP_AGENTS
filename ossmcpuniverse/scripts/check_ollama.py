#!/usr/bin/env python3
"""
Check Ollama setup and verify required models are available.
"""
import os
import sys
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# GPU server models
GPU_MODELS = [
    "gpt-oss:20b",
    "deepseek-r1:14b",
    "gemma3:27b",
    "gemma3:12b",
]

# Local MacBook model (lightweight)
LOCAL_MODELS = [
    "gemma3:4b",
]

# All models
REQUIRED_MODELS = GPU_MODELS + LOCAL_MODELS


def check_ollama_service() -> bool:
    """Check if Ollama service is running."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        return False
    except requests.exceptions.Timeout:
        return False


def get_available_models() -> list:
    """Get list of available models from Ollama."""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        return []
    except Exception as e:
        print(f"Error fetching models: {e}")
        return []


def test_model_generation(model_name: str) -> bool:
    """Test basic generation with a model."""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": "Say 'Hello' in one word."}],
                "stream": False,
                "options": {"num_predict": 10}
            },
            timeout=300
        )
        if response.status_code == 200:
            data = response.json()
            message = data.get("message", {})
            # Check both content and thinking fields (for reasoning models like deepseek-r1)
            content = message.get("content", "")
            thinking = message.get("thinking", "")
            return len(content) > 0 or len(thinking) > 0
        else:
            print(f"Status code: {response.status_code}, Response: {response.text}")
        return False
    except Exception as e:
        print(f"  Error testing {model_name}: {e}")
        return False


def main():
    print("=" * 60)
    print("OSS MCP Universe - Ollama Setup Check")
    print("=" * 60)
    print()

    # Check Ollama URL
    print(f"Ollama URL: {OLLAMA_URL}")
    print()

    # Check if Ollama service is running
    print("1. Checking Ollama service...")
    if not check_ollama_service():
        print("   [FAIL] Ollama service is not running!")
        print(f"   Please start Ollama with: ollama serve")
        sys.exit(1)
    print("   [OK] Ollama service is running")
    print()

    # Get available models
    print("2. Checking available models...")
    available_models = get_available_models()
    print(f"   Found {len(available_models)} model(s):")
    for model in available_models:
        print(f"     - {model}")
    print()

    # Check required models
    print("3. Checking required models...")
    missing_models = []
    for model in REQUIRED_MODELS:
        # Check both exact match and base name match
        model_base = model.split(":")[0]
        found = any(
            m == model or m.startswith(f"{model_base}:")
            for m in available_models
        )
        if found:
            print(f"   [OK] {model}")
        else:
            print(f"   [MISSING] {model}")
            missing_models.append(model)
    print()

    if missing_models:
        print("4. Missing models - please pull them:")
        for model in missing_models:
            print(f"   ollama pull {model}")
        print()

    # Test generation with available required models
    print("5. Testing model generation...")
    available_required = [m for m in REQUIRED_MODELS if m not in missing_models]
    for model in available_required:
        print(f"   Testing {model}...", end=" ", flush=True)
        if test_model_generation(model):
            print("[OK]")
        else:
            print("[FAIL]")
    print()

    # Summary
    print("=" * 60)
    if missing_models:
        print(f"STATUS: {len(missing_models)} model(s) missing")
        print("Please pull the missing models before running benchmarks.")
        sys.exit(1)
    else:
        print("STATUS: All checks passed!")
        print("You can now run benchmarks.")
        sys.exit(0)


if __name__ == "__main__":
    main()
