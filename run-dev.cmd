@echo off
setlocal

cd /d "%~dp0"
set PORT=8001

if not "%~1"=="" (
  set PORT=%~1
)

if not exist ".venv\Scripts\python.exe" (
  echo Creating virtual environment...
  python -m venv .venv
)

netstat -ano | findstr /R /C:":%PORT% .*LISTENING" >nul
if %ERRORLEVEL% EQU 0 (
  echo FastAPI already appears to be running on http://127.0.0.1:%PORT%
  echo Open docs: http://127.0.0.1:%PORT%/docs
  echo To use another port, run: run-dev.cmd 8010
  exit /b 0
)

echo Installing backend packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt

echo Starting FastAPI on http://127.0.0.1:%PORT%
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT% --reload

endlocal
