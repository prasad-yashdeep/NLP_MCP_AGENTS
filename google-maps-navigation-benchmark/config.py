"""
Configuration file for Google Maps API and CrewAI setup.
Add your Google Maps API key here.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# Google Maps API Configuration
# TODO: Add your Google Maps API key here
# You can either:
# 1. Set it as an environment variable: GOOGLE_MAPS_API_KEY=your_key_here
# 2. Or replace the placeholder below with your actual API key

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY", "YOUR_GOOGLE_MAPS_API_KEY_HERE")

# Verify API key is set
if GOOGLE_MAPS_API_KEY == "YOUR_GOOGLE_MAPS_API_KEY_HERE":
    print("WARNING: Please set your GOOGLE_MAPS_API_KEY in config.py or as an environment variable")

# CrewAI Configuration - Using Ollama with Gemma 2 models
# Options: "gemma2:9b", "gemma2:2b", or other Ollama model names
CREWAI_MODEL = os.getenv("CREWAI_MODEL", "gemma2:9b")  # Using local Gemma 2 9B model
CREWAI_TEMPERATURE = float(os.getenv("CREWAI_TEMPERATURE", "0.7"))

# Ollama Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = CREWAI_MODEL  # Alias for clarity

# Benchmark Configuration
TASKS_DIR = "tasks"
OUTPUT_DIR = "results"

