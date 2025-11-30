#!/usr/bin/env python3
"""
Run a single MCP Universe benchmark with a specific Ollama model.

Usage:
    python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation
    python scripts/run_benchmark.py --model deepseek-r1-14b --benchmark browser_automation
    python scripts/run_benchmark.py --model gemma3-12b --benchmark financial_analysis

    # Run only first 5 tasks
    python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation --limit 5
"""
import os
import sys
import argparse
import asyncio
from datetime import datetime
from pathlib import Path

# Add MCP-Universe to path
MCP_UNIVERSE_PATH = Path(__file__).parent.parent.parent / "MCP-Universe"
sys.path.insert(0, str(MCP_UNIVERSE_PATH))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")

# Import MCP Universe components
from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.benchmark.report import BenchmarkReport
from mcpuniverse.tracer.collectors.file import FileCollector
from mcpuniverse.callbacks.handlers.vprint import get_vprint_callbacks

# Available models and benchmarks
# GPU server models: gpt-oss-20b, deepseek-r1-14b, gemma3-12b
# Local MacBook model: gemma3-4b
MODELS = ["gpt-oss-20b", "deepseek-r1-14b", "gemma3-12b", "gemma3-4b"]
BENCHMARKS = [
    "location_navigation",
    "browser_automation",
    "financial_analysis",
    "repository_management",
    "web_search"
]


def get_config_path(model: str, benchmark: str) -> Path:
    """Get the path to the benchmark config file."""
    config_dir = Path(__file__).parent.parent / "configs"
    config_file = config_dir / f"{benchmark}_{model}.yaml"
    return config_file


def get_log_path(model: str, benchmark: str, limit: int = None) -> Path:
    """Get the path for the log file."""
    log_dir = Path(__file__).parent.parent / "logs"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_suffix = f"_limit{limit}" if limit else ""
    log_file = log_dir / f"{benchmark}_{model}{limit_suffix}_{timestamp}.log"
    return log_file


def get_results_path(model: str, benchmark: str, limit: int = None) -> Path:
    """Get the path for the results file."""
    results_dir = Path(__file__).parent.parent / "results"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    limit_suffix = f"_limit{limit}" if limit else ""
    results_file = results_dir / f"{benchmark}_{model}{limit_suffix}_{timestamp}.json"
    return results_file


async def run_benchmark(model: str, benchmark: str, limit: int = None, verbose: bool = True):
    """Run a benchmark with the specified model."""
    config_path = get_config_path(model, benchmark)
    log_path = get_log_path(model, benchmark, limit)
    results_path = get_results_path(model, benchmark, limit)

    # Validate config exists
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    print("=" * 60)
    print(f"OSS MCP Universe Benchmark Runner")
    print("=" * 60)
    print(f"Model:     {model}")
    print(f"Benchmark: {benchmark}")
    print(f"Config:    {config_path}")
    print(f"Log:       {log_path}")
    print(f"Results:   {results_path}")
    if limit:
        print(f"Limit:     {limit} tasks")
    print("=" * 60)
    print()

    # Set up trace collector
    trace_collector = FileCollector(log_file=str(log_path))

    # Set up callbacks
    callbacks = get_vprint_callbacks() if verbose else None

    # Create and run benchmark
    print("Starting benchmark...")
    print()

    try:
        # Change to MCP-Universe directory for config resolution
        original_cwd = os.getcwd()
        os.chdir(MCP_UNIVERSE_PATH / "mcpuniverse" / "benchmark" / "configs")

        benchmark_runner = BenchmarkRunner(str(config_path))

        # Apply task limit if specified
        original_task_count = 0
        if limit and benchmark_runner._benchmark_configs:
            for bc in benchmark_runner._benchmark_configs:
                original_task_count = len(bc.tasks)
                bc.tasks = bc.tasks[:limit]
                print(f"Limited tasks: {original_task_count} -> {len(bc.tasks)}")
                print()

        results = await benchmark_runner.run(
            trace_collector=trace_collector,
            callbacks=callbacks
        )

        os.chdir(original_cwd)

        # Generate report
        print()
        print("=" * 60)
        print("Benchmark Complete - Generating Report")
        print("=" * 60)

        report = BenchmarkReport(benchmark_runner, trace_collector=trace_collector)
        report.dump()

        # Save results summary
        import json
        actual_task_count = len(benchmark_runner._benchmark_configs[0].tasks) if benchmark_runner._benchmark_configs else 0
        results_summary = {
            "model": model,
            "benchmark": benchmark,
            "timestamp": datetime.now().isoformat(),
            "config_path": str(config_path),
            "log_path": str(log_path),
            "task_limit": limit,
            "tasks_run": actual_task_count,
            "original_task_count": original_task_count if limit else actual_task_count,
            "num_results": len(results) if results else 0,
        }

        with open(results_path, "w") as f:
            json.dump(results_summary, f, indent=2, default=str)

        print(f"\nResults saved to: {results_path}")

        return results

    except Exception as e:
        print(f"\nError running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Run MCP Universe benchmark with Ollama model"
    )
    parser.add_argument(
        "--model", "-m",
        required=True,
        choices=MODELS,
        help=f"Model to use: {', '.join(MODELS)}"
    )
    parser.add_argument(
        "--benchmark", "-b",
        required=True,
        choices=BENCHMARKS,
        help=f"Benchmark to run: {', '.join(BENCHMARKS)}"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of tasks to run (e.g., --limit 5 runs only first 5 tasks)"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress verbose output"
    )

    args = parser.parse_args()

    # Run the benchmark
    asyncio.run(run_benchmark(
        model=args.model,
        benchmark=args.benchmark,
        limit=args.limit,
        verbose=not args.quiet
    ))


if __name__ == "__main__":
    main()
