#!/bin/bash

# Directory for containers
CONTAINER_DIR="/scratch/yp2693/NLP_MCP_AGENTS/containers"
mkdir -p "$CONTAINER_DIR"

# Ollama Image Path
OLLAMA_SIF="$CONTAINER_DIR/ollama.sif"

echo "Setting up Ollama in $CONTAINER_DIR..."

if [ -f "$OLLAMA_SIF" ]; then
    echo "Ollama Singularity image already exists at $OLLAMA_SIF"
    read -p "Do you want to pull it again? (y/N): " confirm
    if [[ $confirm != [yY] ]]; then
        echo "Skipping pull."
        exit 0
    fi
fi

echo "Pulling Ollama image from Docker Hub..."
singularity pull --force "$OLLAMA_SIF" docker://ollama/ollama:latest

if [ $? -eq 0 ]; then
    echo "Successfully pulled Ollama image to $OLLAMA_SIF"
else
    echo "Failed to pull Ollama image."
    exit 1
fi

