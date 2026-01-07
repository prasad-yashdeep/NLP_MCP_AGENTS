#!/bin/bash
# Setup script for OSS MCP Universe on RHEL GPU Server

echo "============================================================"
echo "Setting up OSS MCP Universe Environment"
echo "============================================================"

# Activate conda environment
echo "Activating conda environment..."
eval "$(conda shell.bash hook)"
conda activate /scratch/yp2693/NLP_MCP_AGENTS/penv
if [ $? -eq 0 ]; then
    echo "✓ Conda environment activated: penv"
else
    echo "✗ Failed to activate conda environment"
    exit 1
fi

# Load Node.js module (required for google-maps, playwright, etc.)
echo "Loading Node.js module..."
module load node/22.9.0

# Verify Node.js is loaded
if command -v node &> /dev/null; then
    echo "✓ Node.js $(node --version) loaded"
    echo "✓ npm $(npm --version) loaded"
    echo "✓ npx available"
else
    echo "✗ Failed to load Node.js module"
    exit 1
fi

# Check Ollama service
echo ""
echo "Checking Ollama service..."
if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✓ Ollama service is running"
else
    echo "✗ Ollama service is NOT running"
    echo "  Start it with: ollama serve"
fi

# List available models
echo ""
echo "Available Ollama models:"
ollama list 2>/dev/null || echo "  (Ollama not in PATH or not installed)"

echo ""
echo "============================================================"
echo "Environment ready! You can now run benchmarks."
echo "============================================================"
echo ""
echo "Available benchmarks:"
echo "  ✓ location_navigation (google-maps)"
echo "  ✓ browser_automation (playwright)"
echo "  ✓ financial_analysis (yfinance, calculator)"
echo "  ✗ repository_management (requires Docker - not available)"
echo "  ✓ web_search (google-search)"
echo ""
echo "Quick start:"
echo "  python scripts/run_benchmark.py --model gpt-oss-20b --benchmark location_navigation --limit 5"
echo ""
