#!/bin/bash

BASE_DIR="/scratch/yp2693/NLP_MCP_AGENTS"
INIT_FILE="$BASE_DIR/ollama_init.sh"

if [ ! -f "$INIT_FILE" ]; then
    echo "Error: Ollama server info not found at $INIT_FILE"
    echo "Is the server running?"
    exit 1
fi

source "$INIT_FILE"

# Use the correct SINGULARITYENV to ensure the client talks to the right port
export SINGULARITYENV_OLLAMA_HOST="$OLLAMA_HOST"

# Run the command inside the instance
singularity exec instance://"$OLLAMA_INSTANCE" ollama "$@"

