@echo off
rem One command for Windows: sets up on the first run, then starts the Aimodel app.
rem   start.bat           this PC only
rem   start.bat --phone   let your phone on the same Wi-Fi connect (prints a link)
cd /d "%~dp0"
where python >nul 2>nul || (echo Python 3 is needed: install it from python.org & exit /b 1)
if not exist .venv python -m venv .venv
call .venv\Scripts\activate.bat
python -c "import numpy" 2>nul || pip install -q -r requirements.txt
set ARGS=%*
set ARGS=%ARGS:--phone=--host 0.0.0.0%
python -m aimodel.server %ARGS%
