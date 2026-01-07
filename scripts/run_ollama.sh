#!/bin/bash

# Configuration
BASE_DIR="/scratch/yp2693/NLP_MCP_AGENTS"
CONTAINER_DIR="$BASE_DIR/containers"
OLLAMA_SIF="$CONTAINER_DIR/ollama.sif"
INSTANCE_NAME="ollama_server_$(whoami)_$$"

# Ensure image exists
if [ ! -f "$OLLAMA_SIF" ]; then
    echo "Error: Ollama image not found at $OLLAMA_SIF"
    echo "Please run scripts/setup_ollama.sh first."
    exit 1
fi

# Find a free port
# get_free_port() {
#     python -c 'import socket; s=socket.socket(); s.bind(("", 0)); print(s.getsockname()[1]); s.close()'
# }

# export OLLAMA_PORT=$(get_free_port)
export OLLAMA_PORT=11434
export OLLAMA_HOST="127.0.0.1:$OLLAMA_PORT"

echo "Starting Ollama on port $OLLAMA_PORT..."

# Set environment variables for the container
# We use SINGULARITYENV_ prefix to pass them into the container
export SINGULARITYENV_OLLAMA_HOST="0.0.0.0:$OLLAMA_PORT"
export SINGULARITYENV_OLLAMA_MODELS="$BASE_DIR/ollama/models"
export SINGULARITYENV_OLLAMA_KEEP_ALIVE="24h" # Keep models loaded for longer

# Bind necessary paths
# We bind the scratch directory to itself so paths remain consistent
BIND_ARGS="-B /scratch/yp2693/NLP_MCP_AGENTS:/scratch/yp2693/NLP_MCP_AGENTS"

# Start the instance
# --nv enables GPU support
echo "Launching Singularity instance..."
singularity instance start --nv $BIND_ARGS "$OLLAMA_SIF" "$INSTANCE_NAME"

if [ $? -ne 0 ]; then
    echo "Failed to start Singularity instance."
    exit 1
fi

# Start the server inside the instance
echo "Starting Ollama server..."
singularity exec instance://"$INSTANCE_NAME" ollama serve > "$BASE_DIR/ollama/server_$OLLAMA_PORT.log" 2>&1 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for Ollama server to be ready..."
for i in {1..30}; do
    if curl -s "http://127.0.0.1:$OLLAMA_PORT" > /dev/null; then
        echo "Ollama server is running!"
        echo "API URL: http://127.0.0.1:$OLLAMA_PORT"
        
        # Write connection info to a file for easy sourcing
        INIT_FILE="$BASE_DIR/ollama_init.sh"
        echo "export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT" > "$INIT_FILE"
        echo "export OLLAMA_URL=http://localhost:$OLLAMA_PORT" >> "$INIT_FILE"
        echo "export OLLAMA_INSTANCE=$INSTANCE_NAME" >> "$INIT_FILE"
        echo "export OLLAMA_SIF=$OLLAMA_SIF" >> "$INIT_FILE"
        
        echo "Connection info saved to $INIT_FILE"
        echo "To connect from another terminal, run:"
        echo "  source $INIT_FILE"
        break
    fi
    sleep 1
done

# Trap to cleanup on exit
cleanup() {
    echo "Stopping Ollama server and instance..."
    kill $SERVER_PID 2>/dev/null
    singularity instance stop "$INSTANCE_NAME"
    echo "Done."
}
trap cleanup EXIT INT TERM

# Keep script running
echo "Press Ctrl+C to stop the server."
wait $SERVER_PID

