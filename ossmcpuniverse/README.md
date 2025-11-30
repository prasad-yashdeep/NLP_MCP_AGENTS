# OSS MCP Universe - Ollama Benchmarks

Run MCP Universe benchmarks with local Ollama models for open-source LLM evaluation.

## Models

### GPU Server Models (Linux)
- **gpt-oss:20b** - GPT-style OSS model
- **deepseek-r1:14b** - DeepSeek R1 reasoning model
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

## Prerequisites

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

## Directory Structure

```
ossmcpuniverse/
├── README.md
├── requirements.txt
├── .env.example
├── .env                    # Your local config (gitignored)
├── .gitignore
├── configs/                # Benchmark YAML configs (15 total)
│   ├── location_navigation_*.yaml
│   ├── browser_automation_*.yaml
│   ├── financial_analysis_*.yaml
│   ├── repository_management_*.yaml
│   └── web_search_*.yaml
├── scripts/
│   ├── check_ollama.py     # Verify Ollama setup
│   ├── run_benchmark.py    # Run single benchmark
│   └── run_all.py          # Run all benchmarks
├── results/                # Benchmark results (JSON)
└── logs/                   # Execution logs
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

- **google_maps** - For location navigation tasks
- **playwright** - For browser automation tasks
- **yfinance** - For financial analysis tasks
- **calculator** - For financial calculations
- **weather** - For multi-server location tasks
- **github** - For repository management tasks
- **google-search** - For web search tasks
- **fetch** - For fetching web content
- **notion** - For multi-server tasks

## Troubleshooting

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
