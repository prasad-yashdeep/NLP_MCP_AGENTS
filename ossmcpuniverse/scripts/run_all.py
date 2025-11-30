#!/usr/bin/env python3
"""
Run all MCP Universe benchmarks with all Ollama models.
This runs a 4x5 matrix of models and benchmarks (20 total runs).

Usage:
    python scripts/run_all.py
    python scripts/run_all.py --models gpt-oss-20b deepseek-r1-14b
    python scripts/run_all.py --benchmarks location_navigation financial_analysis
    python scripts/run_all.py --limit 5  # Run only 5 tasks per benchmark
"""
import os
import sys
import argparse
import asyncio
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

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
ALL_MODELS = ["gpt-oss-20b", "deepseek-r1-14b", "gemma3-12b", "gemma3-4b"]
ALL_BENCHMARKS = [
    "location_navigation",
    "browser_automation",
    "financial_analysis",
    "repository_management",
    "web_search"
]


def get_config_path(model: str, benchmark: str) -> Path:
    """Get the path to the benchmark config file."""
    config_dir = Path(__file__).parent.parent / "configs"
    return config_dir / f"{benchmark}_{model}.yaml"


def get_log_path(model: str, benchmark: str, timestamp: str, limit: int = None) -> Path:
    """Get the path for the log file."""
    log_dir = Path(__file__).parent.parent / "logs"
    limit_suffix = f"_limit{limit}" if limit else ""
    return log_dir / f"{benchmark}_{model}{limit_suffix}_{timestamp}.log"


async def run_single_benchmark(
    model: str,
    benchmark: str,
    timestamp: str,
    limit: int = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """Run a single benchmark and return results."""
    config_path = get_config_path(model, benchmark)
    log_path = get_log_path(model, benchmark, timestamp, limit)

    if not config_path.exists():
        return {
            "model": model,
            "benchmark": benchmark,
            "status": "error",
            "error": f"Config file not found: {config_path}"
        }

    try:
        # Set up trace collector
        trace_collector = FileCollector(log_file=str(log_path))

        # Set up callbacks
        callbacks = get_vprint_callbacks() if verbose else None

        # Change to MCP-Universe directory for config resolution
        original_cwd = os.getcwd()
        os.chdir(MCP_UNIVERSE_PATH / "mcpuniverse" / "benchmark" / "configs")

        # Run benchmark
        start_time = datetime.now()
        benchmark_runner = BenchmarkRunner(str(config_path))

        # Apply task limit if specified
        original_task_count = 0
        actual_task_count = 0
        if benchmark_runner._benchmark_configs:
            original_task_count = len(benchmark_runner._benchmark_configs[0].tasks)
            if limit:
                for bc in benchmark_runner._benchmark_configs:
                    bc.tasks = bc.tasks[:limit]
            actual_task_count = len(benchmark_runner._benchmark_configs[0].tasks)

        results = await benchmark_runner.run(
            trace_collector=trace_collector,
            callbacks=callbacks
        )
        end_time = datetime.now()

        os.chdir(original_cwd)

        # Calculate metrics
        duration = (end_time - start_time).total_seconds()

        return {
            "model": model,
            "benchmark": benchmark,
            "status": "completed",
            "duration_seconds": duration,
            "tasks_run": actual_task_count,
            "original_task_count": original_task_count,
            "task_limit": limit,
            "log_path": str(log_path),
        }

    except Exception as e:
        return {
            "model": model,
            "benchmark": benchmark,
            "status": "error",
            "error": str(e)
        }


async def run_all_benchmarks(
    models: List[str],
    benchmarks: List[str],
    limit: int = None,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """Run all specified benchmarks with all specified models."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = []

    total = len(models) * len(benchmarks)
    current = 0

    for model in models:
        for benchmark in benchmarks:
            current += 1
            limit_info = f" (limit: {limit} tasks)" if limit else ""
            print(f"\n[{current}/{total}] Running {benchmark} with {model}{limit_info}...")

            result = await run_single_benchmark(
                model=model,
                benchmark=benchmark,
                timestamp=timestamp,
                limit=limit,
                verbose=verbose
            )

            results.append(result)

            if result["status"] == "completed":
                tasks_info = f"{result['tasks_run']}/{result['original_task_count']} tasks" if limit else f"{result['tasks_run']} tasks"
                print(f"    Completed in {result['duration_seconds']:.1f}s ({tasks_info})")
            else:
                print(f"    Error: {result.get('error', 'Unknown error')}")

    return results


def print_summary(results: List[Dict[str, Any]]):
    """Print a summary table of all results."""
    print("\n" + "=" * 90)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 90)

    # Header
    print(f"\n{'Model':<20} {'Benchmark':<25} {'Status':<12} {'Tasks':<12} {'Duration':<12}")
    print("-" * 90)

    # Results
    for r in results:
        model = r["model"]
        benchmark = r["benchmark"]
        status = r["status"]
        if status == "completed":
            tasks = f"{r['tasks_run']}/{r['original_task_count']}" if r.get('task_limit') else str(r['tasks_run'])
            duration = f"{r.get('duration_seconds', 0):.1f}s"
        else:
            tasks = "N/A"
            duration = "N/A"
        print(f"{model:<20} {benchmark:<25} {status:<12} {tasks:<12} {duration:<12}")

    print("-" * 90)

    # Summary stats
    completed = sum(1 for r in results if r["status"] == "completed")
    failed = sum(1 for r in results if r["status"] == "error")
    total_duration = sum(r.get("duration_seconds", 0) for r in results)
    total_tasks = sum(r.get("tasks_run", 0) for r in results if r["status"] == "completed")

    print(f"\nTotal Runs: {len(results)} | Completed: {completed} | Failed: {failed}")
    print(f"Total Tasks Run: {total_tasks}")
    print(f"Total Duration: {total_duration:.1f}s ({total_duration/60:.1f}m)")


def save_results(results: List[Dict[str, Any]], output_path: Path):
    """Save results to JSON file."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "results": results,
        "summary": {
            "total": len(results),
            "completed": sum(1 for r in results if r["status"] == "completed"),
            "failed": sum(1 for r in results if r["status"] == "error"),
            "total_tasks_run": sum(r.get("tasks_run", 0) for r in results if r["status"] == "completed"),
            "total_duration_seconds": sum(r.get("duration_seconds", 0) for r in results)
        }
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\nResults saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all MCP Universe benchmarks with Ollama models"
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        choices=ALL_MODELS,
        default=ALL_MODELS,
        help=f"Models to use (default: all)"
    )
    parser.add_argument(
        "--benchmarks", "-b",
        nargs="+",
        choices=ALL_BENCHMARKS,
        default=ALL_BENCHMARKS,
        help=f"Benchmarks to run (default: all)"
    )
    parser.add_argument(
        "--limit", "-l",
        type=int,
        default=None,
        help="Limit number of tasks per benchmark (e.g., --limit 5 runs only first 5 tasks)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show verbose output during benchmarks"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output JSON file path (default: results/run_all_TIMESTAMP.json)"
    )

    args = parser.parse_args()

    print("=" * 90)
    print("OSS MCP Universe - Full Benchmark Suite")
    print("=" * 90)
    print(f"\nModels:     {', '.join(args.models)}")
    print(f"Benchmarks: {', '.join(args.benchmarks)}")
    print(f"Total runs: {len(args.models) * len(args.benchmarks)}")
    if args.limit:
        print(f"Task limit: {args.limit} tasks per benchmark")
    print()

    # Run all benchmarks
    results = asyncio.run(run_all_benchmarks(
        models=args.models,
        benchmarks=args.benchmarks,
        limit=args.limit,
        verbose=args.verbose
    ))

    # Print summary
    print_summary(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        results_dir = Path(__file__).parent.parent / "results"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        limit_suffix = f"_limit{args.limit}" if args.limit else ""
        output_path = results_dir / f"run_all{limit_suffix}_{timestamp}.json"

    save_results(results, output_path)


if __name__ == "__main__":
    main()
