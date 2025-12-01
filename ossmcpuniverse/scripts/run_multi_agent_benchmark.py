#!/usr/bin/env python3
"""
Run the Multi-Agent Financial Analysis Benchmark.

This script demonstrates the multi-agent system for financial analysis
using gemma3:27b as orchestrator and gemma3:4b for specialized sub-agents.

Usage:
    ./run_with_node.sh python scripts/run_multi_agent_benchmark.py --limit 5
    ./run_with_node.sh python scripts/run_multi_agent_benchmark.py --task 1
    ./run_with_node.sh python scripts/run_multi_agent_benchmark.py --all
"""

import os
import sys
import json
import asyncio
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Setup paths
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent
MCP_UNIVERSE_ROOT = PROJECT_ROOT.parent / "MCP-Universe"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(MCP_UNIVERSE_ROOT))

from dotenv import load_dotenv
load_dotenv()

# Import multi-agent system
from agents.financial_manager import FinancialAnalysisManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_task(task_num: int) -> dict:
    """Load a benchmark task by number."""
    task_path = MCP_UNIVERSE_ROOT / "mcpuniverse" / "benchmark" / "configs" / "test" / "financial_analysis" / f"yfinance_task_{task_num:04d}.json"

    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    with open(task_path, 'r') as f:
        return json.load(f)


def get_all_tasks() -> list:
    """Get all available task numbers."""
    tasks_dir = MCP_UNIVERSE_ROOT / "mcpuniverse" / "benchmark" / "configs" / "test" / "financial_analysis"
    task_files = sorted(tasks_dir.glob("yfinance_task_*.json"))
    return [int(f.stem.split('_')[-1]) for f in task_files]


async def run_single_task(manager: FinancialAnalysisManager, task_num: int) -> dict:
    """Run a single benchmark task."""
    task = load_task(task_num)

    logger.info(f"=" * 60)
    logger.info(f"Task {task_num}: {task.get('question', '')[:80]}...")
    logger.info(f"=" * 60)

    question = task.get("question", "")
    expected = task.get("expected_response", {})
    output_format = task.get("output_format", {})

    try:
        # Run the multi-agent analysis
        start_time = datetime.now()
        result = await manager.analyze(question, output_format)
        elapsed = (datetime.now() - start_time).total_seconds()

        # Compare with expected
        correct = False
        if expected:
            # Check if values match (allowing for some tolerance)
            try:
                result_value = float(result.get("total value", 0))
                expected_value = float(expected.get("total value", 0))
                result_return = float(result.get("total percentage return", 0))
                expected_return = float(expected.get("total percentage return", 0))

                value_match = abs(result_value - expected_value) / max(expected_value, 1) < 0.05
                return_match = abs(result_return - expected_return) < 1.0

                correct = value_match and return_match
            except (ValueError, TypeError):
                correct = False

        logger.info(f"Result: {result}")
        logger.info(f"Expected: {expected}")
        logger.info(f"Correct: {correct} (took {elapsed:.2f}s)")

        return {
            "task_num": task_num,
            "question": question,
            "result": result,
            "expected": expected,
            "correct": correct,
            "elapsed": elapsed,
            "error": None
        }

    except Exception as e:
        logger.error(f"Task {task_num} failed: {e}")
        return {
            "task_num": task_num,
            "question": question,
            "result": None,
            "expected": expected,
            "correct": False,
            "elapsed": 0,
            "error": str(e)
        }


async def run_benchmark(
    limit: int = None,
    task_nums: list = None,
    orchestrator_model: str = "gemma3:27b",
    worker_model: str = "gemma3:4b"
):
    """Run the multi-agent benchmark."""

    print("=" * 60)
    print("Multi-Agent Financial Analysis Benchmark")
    print("=" * 60)
    print(f"Orchestrator Model: {orchestrator_model}")
    print(f"Worker Model: {worker_model}")
    print("=" * 60)

    # Initialize manager
    manager = FinancialAnalysisManager(
        orchestrator_model=orchestrator_model,
        worker_model=worker_model
    )

    try:
        await manager.initialize()

        # Determine tasks to run
        all_tasks = get_all_tasks()

        if task_nums:
            tasks_to_run = [t for t in task_nums if t in all_tasks]
        elif limit:
            tasks_to_run = all_tasks[:limit]
        else:
            tasks_to_run = all_tasks

        print(f"Running {len(tasks_to_run)} task(s)...")
        print()

        results = []
        for task_num in tasks_to_run:
            result = await run_single_task(manager, task_num)
            results.append(result)
            print()

        # Summary
        print("=" * 60)
        print("SUMMARY")
        print("=" * 60)

        total = len(results)
        correct = sum(1 for r in results if r["correct"])
        failed = sum(1 for r in results if r["error"])
        total_time = sum(r["elapsed"] for r in results)

        print(f"Total Tasks: {total}")
        print(f"Correct: {correct} ({100*correct/total:.1f}%)")
        print(f"Failed: {failed}")
        print(f"Total Time: {total_time:.2f}s")
        print(f"Avg Time per Task: {total_time/total:.2f}s")
        print()

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = PROJECT_ROOT / "results" / f"multi_agent_benchmark_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)

        with open(results_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "orchestrator_model": orchestrator_model,
                "worker_model": worker_model,
                "total": total,
                "correct": correct,
                "failed": failed,
                "total_time": total_time,
                "results": results
            }, f, indent=2)

        print(f"Results saved to: {results_file}")

        return results

    finally:
        await manager.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Agent Financial Analysis Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run")
    parser.add_argument("--task", type=int, action="append", dest="tasks", help="Specific task number(s) to run")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--orchestrator", type=str, default="gemma3:27b", help="Orchestrator model")
    parser.add_argument("--worker", type=str, default="gemma3:4b", help="Worker model")

    args = parser.parse_args()

    # Determine what to run
    if args.all:
        limit = None
        tasks = None
    elif args.tasks:
        limit = None
        tasks = args.tasks
    elif args.limit:
        limit = args.limit
        tasks = None
    else:
        # Default: run 3 tasks for quick test
        limit = 3
        tasks = None

    # Run benchmark
    asyncio.run(run_benchmark(
        limit=limit,
        task_nums=tasks,
        orchestrator_model=args.orchestrator,
        worker_model=args.worker
    ))


if __name__ == "__main__":
    main()
