@echo off
rem One-click local start on Windows: creates .venv, installs, opens the web UI.
rem   start.bat          UI + local OCR + cloud VLM clients (Claude / GPT / Gemini)
rem   start.bat full     also local HuggingFace VLMs and the YOLO detector (large download)
rem   start.bat update   reinstall after pulling new code
rem API keys: put lines like  ANTHROPIC_API_KEY=sk-...  in a file named .env next to this script.
setlocal
cd /d "%~dp0"

set "EXTRAS=ui,ocr,anthropic,openai,gemini"
if /i "%~1"=="full" set "EXTRAS=all"

rem ---- Python 3.10+ -----------------------------------------------------------
set "PY="
where py >nul 2>nul && py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=py -3"
if not defined PY (
  where python >nul 2>nul && python -c "import sys; sys.exit(sys.version_info < (3, 10))" >nul 2>nul && set "PY=python"
)
if not defined PY (
  echo Python 3.10 or newer is required. Install it with:
  echo     winget install Python.Python.3.12
  echo or from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^), then run this again.
  pause
  exit /b 1
)

rem ---- virtual environment + install (first run, "update" or "full") ------------
if not exist ".venv\Scripts\python.exe" (
  echo Creating the virtual environment in .venv ...
  %PY% -m venv .venv || (echo Could not create .venv & pause & exit /b 1)
)
set "VPY=.venv\Scripts\python.exe"
set "STAMP=.venv\drawid_installed_%EXTRAS:,=_%.txt"
if /i "%~1"=="update" del /q ".venv\drawid_installed_*.txt" >nul 2>nul
if not exist "%STAMP%" (
  echo Installing drawing-identifier [%EXTRAS%] - the first time takes a few minutes ...
  "%VPY%" -m pip install --upgrade pip
  "%VPY%" -m pip install -e ".[%EXTRAS%]" || (echo Installation failed, see the messages above. & pause & exit /b 1)
  "%VPY%" -m drawing_identifier.cli fonts || echo   ^(fonts not downloaded: the synthetic generator will use built-in fonts^)
  echo ok> "%STAMP%"
)

rem ---- API keys from .env (optional) --------------------------------------------
if exist ".env" (
  for /f "usebackq eol=# tokens=1,* delims==" %%a in (".env") do if not "%%b"=="" set "%%a=%%b"
)

rem ---- optional tools -----------------------------------------------------------
"%VPY%" -c "from drawing_identifier.transcription.engines import find_tesseract as f; import sys; sys.exit(0 if f() else 1)" >nul 2>nul || (
  echo.
  echo [optional] Tesseract OCR not found. For the fastest local OCR tier install it with:
  echo     winget install UB-Mannheim.TesseractOCR
)
where ollama >nul 2>nul || (
  echo [optional] Ollama not found. For a free local VLM: https://ollama.com/download then
  echo     ollama pull qwen2.5vl:3b
)
if not defined ANTHROPIC_API_KEY if not defined OPENAI_API_KEY if not defined GEMINI_API_KEY (
  echo [optional] No cloud API key set: put ANTHROPIC_API_KEY=... in .env to use Claude.
)
echo.
echo Starting the UI on http://127.0.0.1:7860  - close this window to stop it.
"%VPY%" -m drawing_identifier.cli ui --open
pause
