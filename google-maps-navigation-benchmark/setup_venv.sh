#!/bin/bash
# Setup script for Linux/Mac to create virtual environment and install dependencies

echo "Creating virtual environment 'NLP'..."
python3 -m venv NLP

echo "Activating virtual environment..."
source NLP/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source NLP/bin/activate"
echo ""
echo "Make sure Ollama is running with Gemma 2 models:"
echo "  ollama serve"
echo "  ollama pull gemma2:9b"
echo "  ollama pull gemma2:2b"
echo ""
echo "Then add your Google Maps API key to config.py or set GOOGLE_MAPS_API_KEY environment variable."

