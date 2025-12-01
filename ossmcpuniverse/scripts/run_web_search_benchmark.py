#!/usr/bin/env python3
"""
Run the Multi-Agent Web Search Benchmark.

This script demonstrates the multi-agent system for web search
using gemma3:27b as orchestrator and gemma3:4b for specialized sub-agents.

Usage:
    ./run_with_node.sh python scripts/run_web_search_benchmark.py --limit 5
    ./run_with_node.sh python scripts/run_web_search_benchmark.py --task 1
    ./run_with_node.sh python scripts/run_web_search_benchmark.py --all
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

# Import multi-agent systems
from agents.web_search_manager import WebSearchManager
from agents.web_search_react import WebSearchReAct

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_task(task_num: int) -> dict:
    """Load a benchmark task by number."""
    task_path = MCP_UNIVERSE_ROOT / "mcpuniverse" / "benchmark" / "configs" / "test" / "web_search" / f"info_search_task_{task_num:04d}.json"

    if not task_path.exists():
        raise FileNotFoundError(f"Task file not found: {task_path}")

    with open(task_path, 'r') as f:
        return json.load(f)


def get_all_tasks() -> list:
    """Get all available task numbers."""
    tasks_dir = MCP_UNIVERSE_ROOT / "mcpuniverse" / "benchmark" / "configs" / "test" / "web_search"
    task_files = sorted(tasks_dir.glob("info_search_task_*.json"))
    return [int(f.stem.split('_')[-1]) for f in task_files]


async def run_single_task(manager: WebSearchManager, task_num: int) -> dict:
    """Run a single benchmark task."""
    task = load_task(task_num)

    logger.info(f"=" * 60)
    logger.info(f"Task {task_num}: {task.get('question', '')[:80]}...")
    logger.info(f"=" * 60)

    question = task.get("question", "")
    output_format = task.get("output_format", {})

    # Extract expected answer from evaluators section
    expected = {}
    if "evaluators" in task and task["evaluators"]:
        evaluator = task["evaluators"][0]
        if "op_args" in evaluator and "correct_answer" in evaluator["op_args"]:
            expected = {"answer": evaluator["op_args"]["correct_answer"]}

    try:
        # Run the multi-agent search
        start_time = datetime.now()
        result = await manager.search(question, output_format)
        elapsed = (datetime.now() - start_time).total_seconds()

        # Compare with expected
        correct = False
        if expected:
            expected_answer = expected.get("answer", "").lower().strip()
            result_answer = result.get("answer", "").lower().strip()

            # Check if answers match (exact or contains)
            correct = (
                expected_answer == result_answer or
                expected_answer in result_answer or
                result_answer in expected_answer
            )

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
        import traceback
        traceback.print_exc()
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
    worker_model: str = "gemma3:4b",
    use_react: bool = False,
    max_iterations: int = 5
):
    """Run the multi-agent web search benchmark."""

    agent_type = "ReAct" if use_react else "Pipeline"
    print("=" * 60)
    print(f"Multi-Agent Web Search Benchmark ({agent_type})")
    print("=" * 60)
    print(f"Orchestrator Model: {orchestrator_model}")
    print(f"Worker Model: {worker_model}")
    if use_react:
        print(f"Max ReAct Iterations: {max_iterations}")
    print("=" * 60)

    # Initialize manager or ReAct agent
    if use_react:
        manager = WebSearchReAct(
            orchestrator_model=orchestrator_model,
            worker_model=worker_model,
            max_iterations=max_iterations
        )
    else:
        manager = WebSearchManager(
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
        results_file = PROJECT_ROOT / "results" / f"web_search_benchmark_{timestamp}.json"
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
    parser = argparse.ArgumentParser(description="Run Multi-Agent Web Search Benchmark")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of tasks to run")
    parser.add_argument("--task", type=int, action="append", dest="tasks", help="Specific task number(s) to run")
    parser.add_argument("--all", action="store_true", help="Run all tasks")
    parser.add_argument("--orchestrator", type=str, default="gemma3:27b", help="Orchestrator model")
    parser.add_argument("--worker", type=str, default="gemma3:4b", help="Worker model")
    parser.add_argument("--react", action="store_true", help="Use ReAct-style agent (Thought-Action-Observation loop)")
    parser.add_argument("--max-iterations", type=int, default=5, help="Max iterations for ReAct agent (default: 5)")

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
        # Default: run 1 task for quick test
        limit = 1
        tasks = None

    # Run benchmark
    asyncio.run(run_benchmark(
        limit=limit,
        task_nums=tasks,
        orchestrator_model=args.orchestrator,
        worker_model=args.worker,
        use_react=args.react,
        max_iterations=args.max_iterations
    ))


if __name__ == "__main__":
    main()
