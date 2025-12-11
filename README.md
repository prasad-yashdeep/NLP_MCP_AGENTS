# Dual Perspectives on LLM Agent Performance: Evaluating Agentic Architectures with MCP

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MCP-Universe](https://img.shields.io/badge/Benchmark-MCP--Universe-green.svg)](https://github.com/SalesforceAIResearch/MCP-Universe)

> **How does agent architecture affect performance when the tool surface is held constant via MCP?**

This repository contains the code, benchmarks, and analysis for our research on comparing single-agent vs. multi-agent architectures for LLM tool use, evaluated on standardized Model Context Protocol (MCP) benchmarks.

<p align="center">
  <img src="docs/architecture_overview.png" alt="Architecture Overview" width="800"/>
</p>

## 📋 Table of Contents

- [Key Findings](#-key-findings)
- [Research Questions](#-research-questions)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Benchmarks](#-benchmarks)
- [Architectures](#-architectures)
- [Results](#-results)
- [Citation](#-citation)
- [Team](#-team)
- [Acknowledgments](#-acknowledgments)

## 🎯 Key Findings

| Finding | Description |
|---------|-------------|
| **Multi-agent matches frontier models** | GPT-OSS 120B multi-agent achieves 42.9% success (same as Gemini 2.5 Pro) with 31× fewer LLM calls |
| **Architecture compensates for model size** | Multi-agent orchestration enables smaller models to match larger single-agent performance |
| **MCP enables fair comparison** | 92.3% tool success rate confirms performance differences are architectural, not tool-related |
| **Exceptional cost efficiency** | CrewAI + Gemma 2 9B achieves 68.2% success at 0.04% of GPT-4.1's cost |

## 🔬 Research Questions

1. **Which agent architecture demonstrates superior performance across multi-domain MCP tasks?**
2. **Does MCP standardization enable fair architecture comparison?**
3. **Which task types benefit most from multi-agent coordination?**
4. **What is the cost-performance tradeoff for local vs. cloud models?**

## 📁 Project Structure

```
NLP_MCP_AGENTS/
├── MCP-Universe/                    # Forked MCP-Universe benchmark
│   └── mcpuniverse/
│       ├── benchmark/
│       │   ├── configs/
│       │   │   └── test/
│       │   │       ├── web_search.yaml
│       │   │       ├── location_navigation.yaml
│       │   │       └── web_search/          # 55 task definitions
│       │   ├── runner.py
│       │   └── report.py
│       ├── agents/
│       │   ├── web_search_react.py          # Web search orchestrator
│       │   ├── query_formulation_agent.py   # Query optimization
│       │   ├── search_execution_agent.py    # Search execution
│       │   ├── content_fetch_agent.py       # Content retrieval
│       │   └── fact_verification_agent.py   # Fact verification
│       └── mcp/
│           └── servers/
│               ├── google_search/           # SerpAPI integration
│               └── fetch/                   # Content fetching
├── crewai_navigation/               # CrewAI location navigation
│   ├── agents/
│   │   ├── orchestrator.py
│   │   ├── route_planning_agent.py
│   │   ├── distance_optimization_agent.py
│   │   ├── time_optimization_agent.py
│   │   └── place_finding_agent.py
│   ├── tools/
│   │   └── google_maps_tools.py     # 8 MCP-wrapped tools
│   └── llm_config.py                # Ollama/Gemma configuration
├── results/                         # Benchmark outputs
│   ├── web_search/
│   │   ├── GPTOSS120B_multiagent.md
│   │   ├── GPTOSS20B_single.md
│   │   └── Gemini2.5_Pro.md
│   └── location_navigation/
│       └── crewai_gemma2_9b.md
├── docs/
│   ├── WEB_SEARCH_ARCHITECTURE.md
│   ├── LOCATION_NAVIGATION_ARCHITECTURE.md
│   └── figures/
└── paper/
    ├── acl_paper.tex
    └── acl_paper.pdf
```

## 🛠 Installation

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) (for local model inference)
- Google Cloud API key (for Maps API)
- SerpAPI key (for web search)

### Setup

```bash
# Clone the repository
git clone https://github.com/prasad-yashdeep/NLP_MCP_AGENTS.git
cd NLP_MCP_AGENTS

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MCP-Universe
cd MCP-Universe
pip install -e .
cd ..

# Install CrewAI components
pip install crewai==0.41.1 crewai-tools

# Pull Ollama models (for local inference)
ollama pull gemma2:9b
ollama pull gemma3:27b
```

### Environment Variables

Create a `.env` file in the project root:

```bash
# API Keys
SERPAPI_KEY=your_serpapi_key
GOOGLE_MAPS_API_KEY=your_google_maps_key
OPENROUTER_API_KEY=your_openrouter_key  # For GPT-OSS models

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434

# MCP Configuration
MCP_SERVER_TIMEOUT=30
```

## 🚀 Quick Start

### Run Web Search Benchmark

```bash
# Single-agent baseline (GPT-OSS 20B)
python -m mcpuniverse.benchmark.runner \
    --config benchmark/configs/test/web_search.yaml \
    --agent single \
    --model gpt-oss-20b

# Multi-agent (GPT-OSS 120B orchestrator)
python -m mcpuniverse.benchmark.runner \
    --config benchmark/configs/test/web_search.yaml \
    --agent multi \
    --model gpt-oss-120b

# Gemini 2.5 Pro baseline
python -m mcpuniverse.benchmark.runner \
    --config benchmark/configs/test/web_search.yaml \
    --agent single \
    --model gemini-2.5-pro
```

### Run Location Navigation Benchmark

```bash
# CrewAI with Gemma 2 9B
cd crewai_navigation
python run_benchmark.py --tasks all --model gemma2:9b

# Run specific task categories
python run_benchmark.py --tasks route_planning --model gemma2:9b
python run_benchmark.py --tasks place_finding --model gemma2:9b
```

### Generate Benchmark Report

```bash
python -m mcpuniverse.benchmark.report \
    --input results/web_search/ \
    --output reports/web_search_summary.md
```

## 📊 Benchmarks

### Web Search (MCP-Universe)

| Task Category | # Tasks | Description |
|--------------|---------|-------------|
| Factual Information | 17 | Single/multi-fact lookup, statistics |
| Comparison & Analysis | 14 | Entity comparison, rankings, trends |
| Research & Synthesis | 14 | Deep research, topic summarization |
| Current Events | 8 | News retrieval, live data |
| Specialized Search | 2 | Local search, product search |

**Example Task:**
```json
{
  "category": "web_search",
  "question": "What is the current population of Tokyo and its GDP?",
  "mcp_servers": ["google-search", "fetch"],
  "evaluators": [
    {"func": "json -> get(population)", "op": "in_range", "value": [13000000, 14500000]}
  ]
}
```

### Location Navigation (MCP-Universe)

| Task Category | # Tasks | Avg Success | Description |
|--------------|---------|-------------|-------------|
| Place Finding | 11 | 72.8% | Location discovery, coordinate search |
| Time Optimization | 9 | 68.4% | Travel time minimization |
| Route Planning | 10 | 67.5% | Multi-city itineraries |
| Distance Optimization | 10 | 67.1% | Midpoint calculation |
| Multi-modal | 5 | 0% | Complex real-time constraints |

## 🏗 Architectures

### Web Search Multi-Agent (OpenAI Agents SDK)

```
┌─────────────────────────────────────────────────────────────┐
│           WebSearchOrchestrator (GPT-OSS 120B)              │
│                  ReAct Loop (max 12 iter)                   │
└─────────────────────────┬───────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    ▼                     ▼                     ▼
┌─────────┐         ┌─────────┐         ┌─────────┐
│ Query   │         │ Search  │         │ Content │
│ Agent   │         │ Agent   │         │ Fetch   │
│ (20B)   │         │ (20B)   │         │ (20B)   │
└────┬────┘         └────┬────┘         └────┬────┘
     │                   │                   │
     └───────────────────┼───────────────────┘
                         ▼
              ┌──────────────────┐
              │   MCP Servers    │
              │ • Google Search  │
              │ • Fetch          │
              └──────────────────┘
```

### Location Navigation Multi-Agent (CrewAI)

```
┌─────────────────────────────────────────────────────────────┐
│            CrewAI Orchestrator (Gemma 2 9B)                 │
│     Task Analysis → Agent Selection → Response Synthesis    │
└─────────────────────────┬───────────────────────────────────┘
                          │
    ┌──────────┬──────────┼──────────┬──────────┐
    ▼          ▼          ▼          ▼          ▼
┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
│ Route  │ │Distance│ │  Time  │ │ Place  │
│Planning│ │  Opt   │ │  Opt   │ │Finding │
└───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
    └──────────┴──────────┴──────────┘
                    │
         ┌──────────▼──────────┐
         │  Google Maps MCP    │
         │  (8 tools)          │
         └─────────────────────┘
```

## 📈 Results

### Web Search Benchmark

| Model (+ Architecture) | Tasks | Success Rate | Avg LLM Calls | Efficiency |
|------------------------|-------|--------------|---------------|------------|
| GPT-OSS 20B (Single) | 7 | **0.0%** | 168.9 | Very Low |
| +Multi-Agent (120B orch.) | 7 | **42.9%** | 5.4 | High |
| Gemini 2.5 Pro (Single) | 7 | **42.9%** | 8.3 | High |

**Key Insight:** Multi-agent achieves same success as Gemini 2.5 Pro with 31× fewer LLM calls than single-agent baseline.

### Location Navigation Benchmark

| Model (+ Framework) | Tasks | Route Plan | Time Opt | Dist Opt | Place Find | Overall |
|---------------------|-------|------------|----------|----------|------------|---------|
| Gemma 2 9B + CrewAI | 45 | 67.5% | 68.4% | 67.1% | 72.8% | **68.2%** |
| GPT-4.1 (baseline) | 45 | 62.5% | 81.1% | 65.2% | 88.8% | **86.7%** |

**Key Insight:** CrewAI achieves 68.2% success at 0.04% of GPT-4.1's cost ($0.05 vs $127.50 per 1K queries).

### MCP Tool Success Rates

| Tool | Calls | Success Rate |
|------|-------|--------------|
| maps_geocode | 234 | 95.7% |
| maps_search_places | 189 | 91.2% |
| maps_directions | 167 | 89.8% |
| maps_distance_matrix | 143 | 93.4% |
| **Average** | **733** | **92.3%** |

## 📄 Citation

If you use this work, please cite:

```bibtex
@article{prasad2025dual,
  title={Dual Perspectives on LLM Agent Performance: Evaluating Agentic Architectures with MCP},
  author={Prasad, Yashdeep and Shetty, Bhumika Dinesh and Gehani, Ronit and Jagtap, Vedant},
  journal={arXiv preprint},
  year={2025}
}
```

## 👥 Team

| Name | Contribution | Contact |
|------|--------------|---------|
| **Yashdeep Prasad** | MCP-Universe setup, evaluation pipeline, analysis | yp2693@nyu.edu |
| **Bhumika Dinesh Shetty** | AutoGen integration, location/navigation experiments | bds9746@nyu.edu |
| **Ronit Gehani** | Playwright/browser-automation tests, error analysis | rg4881@nyu.edu |
| **Vedant Jagtap** | CrewAI setup, OpenRouter routing, metric collation | vsj7589@nyu.edu |

## 🙏 Acknowledgments

- [MCP-Universe](https://github.com/SalesforceAIResearch/MCP-Universe) by Salesforce AI Research
- [CrewAI](https://github.com/joaomdmoura/crewAI) framework
- NYU HPC for computing resources
- NYU NLP research group for feedback

## 📚 References

- Schick et al. (2023). [Toolformer: Language Models Can Learn to Use Tools](https://arxiv.org/abs/2302.04761)
- Yao et al. (2023). [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- Hong et al. (2023). [MetaGPT: Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352)
- Liu et al. (2023). [AgentBench: Evaluating LLMs as Agents](https://arxiv.org/abs/2308.03688)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  <b>Built with ❤️ at New York University</b>
</p>
