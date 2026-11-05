@echo off
title M.A.R.K. SYSTEM
echo Avvio di M.A.R.K. in corso...

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERRORE] Python non trovato nel PATH. Installalo da https://www.python.org/
    pause
    exit /b 1
)

set PYTHONUNBUFFERED=1
set GRADIO_ANALYTICS_ENABLED=False
set GRADIO_NUM_PORTS=100

python -c "import gradio, llama_cpp" >nul 2>&1
if errorlevel 1 (
    echo Installazione dipendenze in corso (solo la prima volta)...
    pip install --upgrade requests
    pip install https://gradio-pypi-previews.s3.amazonaws.com/df032a1f08a08828fced3016b3b4258ae90e70cf/gradio-6.9.0-py3-none-any.whl
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu --force-reinstall
    pip install sentence-transformers faiss-cpu pypdf
    pip install git+https://github.com/EduardoMilani8/generic-audio-transcriber.git
    pip install edge-tts static-ffmpeg
)

echo Avvio M.A.R.K...
python avvia_programma.py
pause