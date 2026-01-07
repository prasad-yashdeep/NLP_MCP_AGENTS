#!/usr/bin/env python3
"""
Test the ReAct-style Location Navigation Agent.

This script tests the ReAct agent on location navigation tasks.

Usage:
    ./run_with_node.sh python scripts/test_location_react.py --task 1
    ./run_with_node.sh python scripts/test_location_react.py --task 2
"""

import os
import sys
import json
import asyncio
import argparse
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
from agents.location_navigation_react import LocationNavigationReAct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_task(task_num: int) -> dict:
    """Load a location navigation task."""
    task_path = (
        MCP_UNIVERSE_ROOT
        / "mcpuniverse"
        / "benchmark"
        / "configs"
        / "test"
        / "location_navigation"
        / f"google_maps_task_{task_num:04d}.json"
    )

    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    with open(task_path, 'r') as f:
        return json.load(f)


async def test_react(task_num: int = 1):
    """Test ReAct agent on a location navigation task."""

    task = load_task(task_num)
    question = task.get("question", "")
    output_format = task.get("output_format", {})

    print("=" * 70)
    print("Testing ReAct Location Navigation Agent")
    print("=" * 70)
    print(f"Task {task_num}")
    print(f"Question: {question[:150]}...")
    print("=" * 70)

    agent = LocationNavigationReAct(
        orchestrator_model="gemma3:27b",
        worker_model="gemma3:4b",
        max_iterations=10
    )

    try:
        await agent.initialize()

        result = await agent.search(question, output_format)

        print("\n" + "=" * 70)
        print("RESULT")
        print("=" * 70)
        print(json.dumps(result, indent=2))
        print("=" * 70)

        # Check if result has required structure
        if isinstance(result, dict):
            has_routes = "routes" in result
            has_origin = "starting_city" in result
            has_dest = "destination_city" in result
            print(f"\nStructure Check:")
            print(f"  Has routes: {has_routes}")
            print(f"  Has starting_city: {has_origin}")
            print(f"  Has destination_city: {has_dest}")
            if has_routes:
                print(f"  Number of routes: {len(result.get('routes', []))}")
        print("=" * 70)

    except Exception as e:
        logger.error(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        await agent.cleanup()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test ReAct Location Navigation Agent")
    parser.add_argument(
        "--task", "-t",
        type=int,
        default=2,
        help="Task number to test (default: 2, the simplest task)"
    )

    args = parser.parse_args()

    print(f"\n⚠️  Note: This test requires Google Maps API key in .env file")
    print(f"⚠️  API rate limits may be hit during testing\n")

    asyncio.run(test_react(args.task))


if __name__ == "__main__":
    main()
