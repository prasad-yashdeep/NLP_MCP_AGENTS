"""
Main benchmark script for Google Maps Navigation tasks.
Loads tasks from tasks/ folder and runs them through CrewAI hierarchical orchestrator.
"""
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from agents import (
    create_orchestrator_agent,
    create_route_planning_agent,
    create_distance_optimization_agent,
    create_time_optimization_agent,
    create_place_finding_agent,
    create_task_for_agent
)
from evaluator import TaskEvaluator
from google_maps_client import GoogleMapsClient
from config import TASKS_DIR, OUTPUT_DIR
from crewai import Crew, Task


def load_task(task_file: Path) -> Dict[str, Any]:
    """Load a task JSON file."""
    with open(task_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_all_tasks(tasks_dir: str = TASKS_DIR) -> List[Dict[str, Any]]:
    """Load all task files from the tasks directory."""
    tasks_path = Path(tasks_dir)
    task_files = sorted(tasks_path.glob("google_maps_task_*.json"))
    
    tasks = []
    for task_file in task_files:
        try:
            task_data = load_task(task_file)
            task_data["task_id"] = task_file.stem
            tasks.append(task_data)
        except Exception as e:
            print(f"Error loading {task_file}: {e}")
    
    return tasks


def categorize_task(task: Dict[str, Any]) -> str:
    """Determine task category based on question content."""
    question = task.get("question", "").lower()
    
    # Route Planning keywords
    if any(kw in question for kw in ["route", "road trip", "cities between", "scenic viewpoints", "rest stops"]):
        return "Route Planning"
    
    # Distance Optimization keywords
    if any(kw in question for kw in ["quarter points", "midpoint", "nearest to", "closest to", "optimal stops"]):
        return "Distance Optimization"
    
    # Time Optimization keywords
    if any(kw in question for kw in ["equidistant", "meeting point", "equal distance", "travel time"]):
        return "Time Optimization"
    
    # Place Finding keywords
    if any(kw in question for kw in ["find", "identify", "location", "south of", "east of", "latitude", "longitude"]):
        return "Place Finding"
    
    # Default to Route Planning
    return "Route Planning"


def create_crew_for_task(task: Dict[str, Any]) -> Crew:
    """Create a CrewAI crew for a specific task."""
    question = task.get("question", "")
    output_format = task.get("output_format", {})
    category = categorize_task(task)
    
    # Create orchestrator
    orchestrator = create_orchestrator_agent()
    
    # Create specialist agents
    route_agent = create_route_planning_agent()
    distance_agent = create_distance_optimization_agent()
    time_agent = create_time_optimization_agent()
    place_agent = create_place_finding_agent()
    
    # Select appropriate specialist based on category
    category_to_agent = {
        "Route Planning": route_agent,
        "Distance Optimization": distance_agent,
        "Time Optimization": time_agent,
        "Place Finding": place_agent
    }
    
    specialist = category_to_agent.get(category, route_agent)
    
    # Create task for specialist
    specialist_task = Task(
        description=f"""
        Category: {category}
        
        Question: {question}
        
        Output Format (JSON):
        {json.dumps(output_format, indent=2)}
        
        Please analyze the question carefully and use Google Maps APIs to generate 
        a response that exactly matches the required output format. Ensure all 
        criteria are met and provide accurate Place IDs, addresses, elevations, 
        and other required information.
        
        Use the available Google Maps tools to:
        1. Geocode addresses to get coordinates
        2. Find directions and routes
        3. Search for places (rest stops, viewpoints, hotels, etc.)
        4. Get place details including ratings and amenities
        5. Calculate distances and elevations
        
        Return ONLY valid JSON matching the output format.
        """,
        agent=specialist,
        expected_output="JSON response matching the specified output format with all required fields populated"
    )
    
    # Create crew
    crew = Crew(
        agents=[orchestrator, specialist],
        tasks=[specialist_task],
        verbose=True
    )
    
    return crew


def run_benchmark(task: Dict[str, Any], evaluator: TaskEvaluator) -> Dict[str, Any]:
    """Run a single benchmark task."""
    task_id = task.get("task_id", "unknown")
    question = task.get("question", "")
    evaluators = task.get("evaluators", [])
    
    print(f"\n{'='*80}")
    print(f"Running Task: {task_id}")
    print(f"Category: {categorize_task(task)}")
    print(f"Question: {question[:100]}...")
    print(f"{'='*80}\n")
    
    result = {
        "task_id": task_id,
        "category": categorize_task(task),
        "question": question,
        "status": "pending",
        "response": None,
        "evaluation": None,
        "error": None,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # Create crew and run
        crew = create_crew_for_task(task)
        response = crew.kickoff()
        
        result["response"] = str(response)
        result["status"] = "completed"
        
        # Evaluate response
        evaluation = evaluator.evaluate(result["response"], evaluators)
        result["evaluation"] = evaluation
        
        # Print results
        print(f"\nResponse: {result['response'][:200]}...")
        print(f"\nEvaluation:")
        print(f"  Passed: {len(evaluation['passed'])}")
        print(f"  Failed: {len(evaluation['failed'])}")
        if evaluation['errors']:
            print(f"  Errors: {len(evaluation['errors'])}")
        
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"Error running task: {e}")
    
    return result


def save_results(results: List[Dict[str, Any]], output_dir: str = OUTPUT_DIR, incremental: bool = False):
    """Save benchmark results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Use same timestamp for incremental saves
    if not hasattr(save_results, 'timestamp'):
        save_results.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    timestamp = save_results.timestamp
    output_file = output_path / f"benchmark_results_{timestamp}.json"
    
    # Save JSON (overwrite for incremental, or create new)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    if incremental:
        print(f"[Saved] Results updated: {output_file} ({len(results)} tasks)")
    else:
        print(f"\nResults saved to: {output_file}")
    
    # Also save/update summary
    summary_file = output_path / f"benchmark_summary_{timestamp}.txt"
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Google Maps Navigation Benchmark Results\n")
        f.write("=" * 80 + "\n\n")
        
        total = len(results)
        completed = sum(1 for r in results if r["status"] == "completed")
        errors = sum(1 for r in results if r["status"] == "error")
        in_progress = sum(1 for r in results if r["status"] == "pending")
        
        f.write(f"Total Tasks: {total}\n")
        f.write(f"Completed: {completed}\n")
        f.write(f"Errors: {errors}\n")
        f.write(f"In Progress: {in_progress}\n\n")
        
        for result in results:
            f.write(f"\nTask: {result['task_id']}\n")
            f.write(f"Status: {result['status']}\n")
            if result.get("evaluation"):
                eval_data = result["evaluation"]
                f.write(f"Passed: {len(eval_data['passed'])}\n")
                f.write(f"Failed: {len(eval_data['failed'])}\n")
                if eval_data.get('errors'):
                    f.write(f"Errors: {len(eval_data['errors'])}\n")
            if result.get("error"):
                f.write(f"Error: {result['error']}\n")
            f.write("-" * 80 + "\n")
    
    if not incremental:
        print(f"Summary saved to: {summary_file}")


def main():
    """Main benchmark execution."""
    print("Google Maps Navigation Benchmark")
    print("=" * 80)
    print(f"Using CrewAI with Ollama (Gemma 2 models)")
    print(f"Loading tasks from: {TASKS_DIR}")
    print("=" * 80)
    
    # Initialize Google Maps client
    try:
        maps_client = GoogleMapsClient()
        print("[OK] Google Maps API client initialized")
    except ValueError as e:
        print(f"[WARNING] {e}")
        maps_client = None
    
    # Initialize evaluator
    evaluator = TaskEvaluator(maps_client)
    
    # Load tasks
    tasks = load_all_tasks()
    print(f"\nLoaded {len(tasks)} tasks")
    
    if not tasks:
        print("No tasks found! Please check the tasks/ directory.")
        return
    
    # Run benchmark with incremental saving
    results = []
    for i, task in enumerate(tasks, 1):
        print(f"\n[{i}/{len(tasks)}] Processing task...")
        result = run_benchmark(task, evaluator)
        results.append(result)
        
        # Save incrementally after each task
        save_results(results, incremental=True)
        
        # Print progress summary
        completed = sum(1 for r in results if r["status"] == "completed")
        errors = sum(1 for r in results if r["status"] == "error")
        print(f"\nProgress: {i}/{len(tasks)} tasks | Completed: {completed} | Errors: {errors}")
    
    # Final save (non-incremental flag for final message)
    save_results(results, incremental=False)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Benchmark Complete!")
    print("=" * 80)
    total = len(results)
    completed = sum(1 for r in results if r["status"] == "completed")
    errors = sum(1 for r in results if r["status"] == "error")
    
    print(f"Total Tasks: {total}")
    print(f"Completed: {completed}")
    print(f"Errors: {errors}")
    
    if completed > 0:
        total_passed = sum(
            len(r["evaluation"]["passed"]) 
            for r in results 
            if r.get("evaluation") and r["status"] == "completed"
        )
        total_failed = sum(
            len(r["evaluation"]["failed"]) 
            for r in results 
            if r.get("evaluation") and r["status"] == "completed"
        )
        print(f"\nTotal Evaluations Passed: {total_passed}")
        print(f"Total Evaluations Failed: {total_failed}")


if __name__ == "__main__":
    main()

