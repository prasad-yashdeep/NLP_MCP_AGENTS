# Google Maps Navigation Benchmark with CrewAI

A comprehensive benchmark system for evaluating AI agents on Google Maps navigation tasks using CrewAI with a hierarchical orchestrator architecture. This project uses local Gemma 2 models via Ollama and integrates with Google Maps APIs for route planning, place finding, and navigation optimization.

## Overview

This benchmark evaluates AI agents on 35 diverse navigation tasks across 4 categories:
- **Route Planning** (10 tasks): Multi-city routes with stops and viewpoints
- **Distance Optimization** (16 tasks): Finding optimal stops along routes
- **Time Optimization** (4 tasks): Equidistant meeting points
- **Place Finding** (5 tasks): Location discovery with geographic constraints

## Architecture

The system uses a **hierarchical orchestrator architecture**:
- **Orchestrator Agent**: Analyzes questions and delegates to specialist agents
- **Specialist Agents**: 
  - Route Planning Specialist
  - Distance Optimization Specialist
  - Time Optimization Specialist
  - Place Finding Specialist

Each agent has access to 8 Google Maps tools:
- Geocoding & Reverse Geocoding
- Directions & Routes
- Places Search & Nearby Search
- Place Details
- Distance Matrix
- Elevation Data

## Prerequisites

1. **Python 3.10+**
2. **Ollama** installed and running locally
3. **Gemma 2 models** pulled in Ollama:
   ```bash
   ollama pull gemma2:9b
   ollama pull gemma2:2b
   ```
4. **Google Maps API Key** with the following APIs enabled:
   - Places API
   - Directions API
   - Geocoding API
   - Distance Matrix API

## Quick Start

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd google-maps-navigation-benchmark
```

### 2. Set Up Virtual Environment

**Windows:**
```bash
setup_venv.bat
```

**Linux/Mac:**
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

**Manual Setup (if scripts fail):**
```bash
# Create virtual environment
python -m venv NLP

# Activate (Windows)
NLP\Scripts\activate.bat

# Activate (Linux/Mac)
source NLP/bin/activate

# Upgrade pip first (IMPORTANT to avoid pip errors)
python -m pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key

Edit `config.py` and replace `YOUR_GOOGLE_MAPS_API_KEY_HERE` with your actual API key:

```python
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "your_actual_api_key_here")
```

Or set it as an environment variable:
```bash
# Windows
set GOOGLE_MAPS_API_KEY=your_actual_api_key_here

# Linux/Mac
export GOOGLE_MAPS_API_KEY=your_actual_api_key_here
```

### 4. Ensure Ollama is Running

```bash
# Start Ollama server (if not already running)
ollama serve

# Verify models are available
ollama list
```

### 5. Run the Benchmark

```bash
# Activate virtual environment first
# Windows: NLP\Scripts\activate.bat
# Linux/Mac: source NLP/bin/activate

python benchmark.py
```

## Troubleshooting Installation Issues

### Common Pip Installation Errors

If you encounter pip installation errors when cloning the repository, follow these steps:

1. **Always upgrade pip first:**
   ```bash
   python -m pip install --upgrade pip
   ```

2. **Use verbose mode to see what's happening:**
   ```bash
   pip install -r requirements.txt -v
   ```

3. **If packages fail to install, try installing individually:**
   ```bash
   pip install crewai
   pip install googlemaps
   pip install python-dotenv
   pip install pydantic
   pip install langchain
   pip install langchain-community
   pip install litellm
   pip install ollama
   ```

4. **For Windows PowerShell issues:**
   - Use Command Prompt instead of PowerShell
   - Or run: `python -m pip install -r requirements.txt` instead of `pip install`

5. **Clear pip cache if needed:**
   ```bash
   pip cache purge
   pip install --no-cache-dir -r requirements.txt
   ```

6. **If virtual environment activation fails:**
   - Make sure you're using the correct Python version (3.10+)
   - Try recreating the virtual environment: `python -m venv NLP --clear`

### Model Configuration Issues

If you see errors about "Model gpt-4.1-mini not found":
- This is normal - CrewAI uses OpenAI-compatible API format
- The system is configured to use Ollama, not OpenAI
- Make sure `llm_config.py` is imported before CrewAI agents are created
- Verify Ollama is running: `ollama ps`

### Google Maps API Issues

If you get API key errors:
- Verify your API key is correct in `config.py`
- Check that all required APIs are enabled in Google Cloud Console
- Ensure billing is enabled (Google Maps APIs require billing)
- Check API quotas haven't been exceeded

## Project Structure

```
google-maps-navigation-benchmark/
├── agents.py              # CrewAI agent definitions
├── benchmark.py           # Main benchmark execution script
├── config.py              # Configuration (API keys, model settings)
├── evaluator.py           # Task evaluation logic
├── google_maps_client.py  # Google Maps API wrapper
├── llm_config.py          # Ollama/LiteLLM configuration
├── requirements.txt       # Python dependencies
├── setup_venv.bat         # Windows setup script
├── setup_venv.sh          # Linux/Mac setup script
├── tasks/                 # 35 benchmark task JSON files
│   ├── google_maps_task_0001.json
│   ├── google_maps_task_0002.json
│   └── ...
├── results/              # Benchmark results (created at runtime)
└── README.md             # This file
```

## Configuration

### Model Selection

Edit `config.py` to change the model:

```python
# Options: "gemma2:9b", "gemma2:2b", or other Ollama models
CREWAI_MODEL = os.getenv("CREWAI_MODEL", "gemma2:9b")
```

### Ollama Configuration

Default Ollama URL is `http://localhost:11434`. To change:

```python
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
```

## Results

Results are saved incrementally after each task in the `results/` directory:
- `benchmark_results_YYYYMMDD_HHMMSS.json`: Full results in JSON format
- `benchmark_summary_YYYYMMDD_HHMMSS.txt`: Human-readable summary

Each result includes:
- Task ID and category
- Question and response
- Evaluation results (passed/failed/errors)
- Timestamp

## Development

### Adding New Tasks

Add new task JSON files to the `tasks/` directory following the format:
```json
{
    "category": "Route Planning",
    "question": "Your question here...",
    "output_format": { ... },
    "evaluators": [ ... ]
}
```

### Customizing Agents

Edit `agents.py` to modify agent roles, goals, or backstories.

### Adding New Tools

Add new Google Maps tools in `agents.py` using the `@tool` decorator.

## Notes

- **Incremental Saving**: Results are saved after each task to prevent data loss
- **Unicode Support**: Windows encoding issues are handled automatically
- **Error Handling**: Tool parameter validation handles various input formats from agents
- **No OpenAI Required**: This project uses local Ollama models, not OpenAI

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

[Add your license here]

## Acknowledgments

- CrewAI for the agent framework
- Ollama for local LLM hosting
- Google Maps Platform for navigation APIs

## Support

For issues or questions, please open an issue on GitHub.

---

**Important**: Remember to never commit your actual Google Maps API key to version control. Always use environment variables or keep `config.py` in `.gitignore` for local development.

