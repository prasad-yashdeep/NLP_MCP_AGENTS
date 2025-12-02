@echo off
REM Setup script for Windows to create virtual environment and install dependencies

echo Creating virtual environment 'NLP'...
python -m venv NLP

echo Activating virtual environment...
call NLP\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing requirements...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To activate the virtual environment, run:
echo   NLP\Scripts\activate.bat
echo.
echo Make sure Ollama is running with Gemma 2 models:
echo   ollama serve
echo   ollama pull gemma2:9b
echo   ollama pull gemma2:2b
echo.
echo Then add your Google Maps API key to config.py or set GOOGLE_MAPS_API_KEY environment variable.
echo.
pause

