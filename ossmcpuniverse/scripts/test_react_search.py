#!/usr/bin/env python3
"""
Test the ReAct-style Web Search Agent.

This script tests the ReAct agent on a single task.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
MCP_UNIVERSE_ROOT = PROJECT_ROOT.parent / "MCP-Universe"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MCP_UNIVERSE_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import ReAct agent
from agents.web_search_react import WebSearchReAct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_react():
    """Test ReAct agent on Task 1."""

    question = """I'm looking for someone based on the clues below:
- Score 16 goals in 2024-25 season
- Score 1 goal in UEFA Champions League 2024-25 season
- Score 11 goals in 2021-22 season
- Score 2 goals in the EFL Cup of 2020-21 season."""

    expected_answer = "Ollie Watkins"

    print("=" * 70)
    print("Testing ReAct Web Search Agent")
    print("=" * 70)
    print(f"Question: {question}")
    print(f"Expected Answer: {expected_answer}")
    print("=" * 70)

    agent = WebSearchReAct(
        orchestrator_model="gemma3:27b",
        worker_model="gemma3:4b",
        max_iterations=5
    )

    try:
        await agent.initialize()

        result = await agent.search(question)

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(f"Answer: {result['answer']}")
        print(f"Expected: {expected_answer}")
        print(f"Correct: {result['answer'].lower() == expected_answer.lower()}")
        print("=" * 70)

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(test_react())
