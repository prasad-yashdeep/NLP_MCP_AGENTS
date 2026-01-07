#!/bin/bash
# Wrapper script to run benchmarks with conda env and Node.js module loaded
# This ensures npx is available for MCP servers like google-maps and playwright

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate /scratch/yp2693/NLP_MCP_AGENTS/penv

# Load Node.js module
module load node/22.9.0

# Execute the Python script with all arguments passed through
exec "$@"
