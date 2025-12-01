# OSS MCP Universe - Ollama Benchmarks

Run MCP Universe benchmarks with local Ollama models for open-source LLM evaluation.

## Models

### GPU Server Models (Linux)
- **gpt-oss:20b** - GPT-style OSS model
- **deepseek-r1:14b** - DeepSeek R1 reasoning model
- **gemma3:27b** - Google Gemma 3 model (27B)
- **gemma3:12b** - Google Gemma 3 model (12B)

### Local Testing Model (MacBook)
- **gemma3:4b** - Google Gemma 3 model (4B) - lightweight for local testing

## Benchmarks

| Benchmark | Tasks | Description |
|-----------|-------|-------------|
| Location Navigation | 45 | Google Maps navigation and multi-server tasks |
| Browser Automation | 39 | Playwright web automation tasks |
| Financial Analysis | 40 | YFinance stock analysis with calculator |
| Repository Management | 33 | GitHub repository management tasks |
| Web Search | 55 | Google search and information retrieval tasks |

## Multi-Agent Systems

This project includes advanced **ReAct-based multi-agent architectures** for complex benchmarks:

### 🗺️ Location Navigation ReAct System
- **Architecture**: Orchestrator (gemma3:27b) + 4 specialized workers (gemma3:4b)
- **Workers**: Route Planning, Place Search, Elevation, Data Synthesis
- **Pattern**: Thought-Action-Observation loop with state management
- **Integration**: Google Maps MCP server (7 tools)
- **Documentation**: See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture diagrams

### 🔍 Web Search ReAct System
- **Architecture**: Orchestrator (gemma3:27b) + 3 specialized workers (gemma3:4b)
- **Workers**: Search, Fetch, Synthesis
- **Pattern**: Iterative fact-finding with verification
- **Integration**: Google Search + Fetch MCP servers

### 💰 Financial Analysis Multi-Agent
- **Architecture**: Orchestrator + 3 specialized workers
- **Workers**: Data Agent (YFinance), Calculator Agent, Formatter
- **Pattern**: Task decomposition with tool orchestration
- **Integration**: YFinance + Calculator MCP servers

**Key Benefits**:
- **70% token reduction** vs single large model
- **Specialized expertise** per agent
- **Transparent reasoning** with ReAct pattern
- **Error recovery** through iterative refinement

## Prerequisites

### RHEL GPU Server Setup (REQUIRED)

If you're on the RHEL GPU server, you **must** activate the conda environment and load the Node.js module before running benchmarks:

```bash
# Activate conda environment
conda activate /scratch/yp2693/NLP_MCP_AGENTS/penv

# Load Node.js module (required for google-maps and playwright MCP servers)
module load node/22.9.0

# Or use the setup script to do both
source setup_env.sh
```

Add this to your `~/.bashrc` to load automatically:
```bash
echo "conda activate /scratch/yp2693/NLP_MCP_AGENTS/penv" >> ~/.bashrc
echo "module load node/22.9.0" >> ~/.bashrc
```

**Note:** The `repository_management` benchmark requires Docker (not available on this server) and will be skipped.

### 1. Install Ollama

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.com/install.sh | sh
```

### 2. Start Ollama Service

```bash
ollama serve
```

### 3. Pull Required Models

**For local MacBook testing (lightweight):**
```bash
ollama pull gemma3:4b
```

**For GPU server (full benchmarks):**
```bash
ollama pull gpt-oss:20b
ollama pull deepseek-r1:14b
ollama pull gemma3:27b
ollama pull gemma3:12b
```

### 4. Install Dependencies

```bash
# From the ossmcpuniverse directory
pip install -r requirements.txt

# Or install MCP-Universe directly
cd ../MCP-Universe && pip install -e .
```

### 5. Configure Environment

```bash
cp .env.example .env
# Edit .env if Ollama is not running on localhost:11434
```

## Usage

### Quick Start (RHEL GPU Server)

```bash
# 1. Activate conda environment and load Node.js module
conda activate /scratch/yp2693/NLP_MCP_AGENTS/penv
module load node/22.9.0

# 2. Or use the setup script to do both
source setup_env.sh

# 3. Run a quick test (5 tasks)
python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation --limit 5

# 4. Or use the wrapper script (auto-loads everything)
./run_with_node.sh python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation --limit 5
```

### Check Ollama Setup

Before running benchmarks, verify your Ollama setup:

```bash
python scripts/check_ollama.py
```

### Run Single Benchmark

```bash
# Location Navigation with gpt-oss:20b
python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation

# Browser Automation with deepseek-r1:14b
python scripts/run_benchmark.py --model deepseek-r1-14b --benchmark browser_automation

# Financial Analysis with gemma3:12b
python scripts/run_benchmark.py --model gemma3-12b --benchmark financial_analysis

# Repository Management
python scripts/run_benchmark.py --model gpt-oss-20b --benchmark repository_management

# Web Search
python scripts/run_benchmark.py --model deepseek-r1-14b --benchmark web_search
```

### Local MacBook Testing (gemma3:4b)

For quick local testing on MacBook without GPU:

```bash
# Pull the lightweight model first
ollama pull gemma3:4b

# Run a quick test with limited tasks
python scripts/run_benchmark.py --model gemma3-4b --benchmark location_navigation --limit 3

# Run all benchmarks with gemma3:4b (limited tasks for speed)
python scripts/run_all.py --models gemma3-4b --limit 5
```

### Run with Task Limit

Run only a subset of tasks (useful for quick testing):

```bash
# Run only first 5 tasks
python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation --limit 5

# Run only first 10 tasks
python scripts/run_benchmark.py --model deepseek-r1-14b --benchmark web_search --limit 10
```

### Run All Benchmarks

Run the full 4x5 matrix (20 benchmark runs):

```bash
python scripts/run_all.py
```

Run with options:

```bash
# Only specific models
python scripts/run_all.py --models gpt-oss-20b deepseek-r1-14b

# Only specific benchmarks
python scripts/run_all.py --benchmarks location_navigation financial_analysis

# Limit tasks per benchmark (quick testing)
python scripts/run_all.py --limit 5

# With verbose output
python scripts/run_all.py --verbose

# Combine options
python scripts/run_all.py --models gpt-oss-20b --benchmarks location_navigation web_search --limit 3
```

### Test Multi-Agent ReAct Systems

Test the ReAct agents on individual tasks for debugging and development:

```bash
# Test Location Navigation ReAct agent
./run_with_node.sh python scripts/test_location_react.py --task 2

# Test with different tasks
./run_with_node.sh python scripts/test_location_react.py --task 1

# Test Web Search ReAct agent
./run_with_node.sh python scripts/test_react_search.py --task 1
```

**Note**: These test scripts provide detailed logging of the Thought-Action-Observation loop, useful for understanding agent behavior and debugging issues.

## Directory Structure

```
ossmcpuniverse/
├── README.md                           # This file
├── ARCHITECTURE.md                     # Multi-agent system architecture
├── requirements.txt
├── .env.example
├── .env                                # Your local config (gitignored)
├── .gitignore
├── agents/                             # Multi-agent implementations
│   ├── base.py                         # Base OllamaAgent class
│   ├── models.py                       # Pydantic data models
│   ├── tools.py                        # MCP tool wrappers
│   ├── location_navigation_react.py    # Location ReAct orchestrator
│   ├── route_planning_agent.py         # Route planning worker
│   ├── place_search_agent.py           # Place search worker
│   ├── elevation_agent.py              # Elevation worker
│   ├── data_synthesis_agent.py         # Output synthesis worker
│   ├── web_search_react.py             # Web search ReAct orchestrator
│   ├── search_agent.py                 # Search worker
│   ├── fetch_agent.py                  # Fetch worker
│   ├── synthesis_agent.py              # Web synthesis worker
│   ├── financial_manager.py            # Financial orchestrator
│   ├── data_agent.py                   # YFinance worker
│   ├── calculator_agent.py             # Calculator worker
│   ├── formatter_agent.py              # Output formatter
│   └── __init__.py                     # Agent exports
├── configs/                            # Benchmark YAML configs (18 total)
│   ├── location_navigation_*.yaml      # (5 configs)
│   ├── browser_automation_*.yaml       # (3 configs)
│   ├── financial_analysis_*.yaml       # (4 configs)
│   ├── repository_management_*.yaml    # (3 configs)
│   └── web_search_*.yaml               # (3 configs)
├── scripts/
│   ├── check_ollama.py                 # Verify Ollama setup
│   ├── run_benchmark.py                # Run single benchmark
│   ├── run_all.py                      # Run all benchmarks
│   ├── test_location_react.py          # Test location ReAct agent
│   └── test_react_search.py            # Test web search ReAct agent
├── docs/
│   └── location_architecture_simple.txt # ASCII architecture diagram
├── results/                            # Benchmark results (JSON)
└── logs/                               # Execution logs
```

## Results

Results are saved to:
- `results/` - JSON summary files
- `logs/` - Detailed execution logs

Example results file structure:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "results": [
    {
      "model": "gpt-oss-20b",
      "benchmark": "location_navigation",
      "status": "completed",
      "duration_seconds": 120.5,
      "tasks_run": 45,
      "original_task_count": 45,
      "task_limit": null
    }
  ],
  "summary": {
    "total": 15,
    "completed": 15,
    "failed": 0,
    "total_tasks_run": 212,
    "total_duration_seconds": 1800.5
  }
}
```

## MCP Servers Required

The benchmarks require these MCP servers to be configured in MCP-Universe:

| MCP Server | Command | Benchmarks | Available on RHEL? |
|------------|---------|------------|-------------------|
| google-maps | `npx` | location_navigation | ✓ (with module load) |
| playwright | `npx` | browser_automation | ✓ (with module load) |
| yfinance | `python3` | financial_analysis | ✓ |
| calculator | `python3` | financial_analysis | ✓ |
| github | `docker` | repository_management | ✗ (no Docker) |
| google-search | `python3` | web_search | ✓ |

**On RHEL GPU Server:** 4 out of 5 benchmarks work after loading Node.js module

## Troubleshooting

### "The command must be a valid string" Error

```
ValueError: The command must be a valid string
```

**Cause:** Node.js module not loaded (required for google-maps, playwright, etc.)

**Solution:**
```bash
module load node/22.9.0
# Then retry your benchmark command
```

### Ollama Connection Error

```
Error: Ollama service is not running!
```

Solution: Start Ollama with `ollama serve`

### Model Not Found

```
Error: Model 'gpt-oss:20b' not found
```

Solution: Pull the model with `ollama pull gpt-oss:20b`

### Timeout Errors

For large models, you may need to increase the timeout in the Ollama config:

```python
# In mcpuniverse/llm/ollama.py, increase timeout
response = requests.post(url, json=data, timeout=120)  # Increase from 30
```

### Quick Testing

Use `--limit` to run only a few tasks for quick testing:

```bash
# Test with just 3 tasks per benchmark
python scripts/run_all.py --limit 3
```

## License

This project uses MCP-Universe under the Apache 2.0 license.
