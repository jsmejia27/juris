@echo off
title Juris - Philippine Sovereign Legal AI
color 0B
cd /d "%~dp0"

echo =====================================================================
echo          JURIS - SOVEREIGN PHILIPPINE LEGAL AI & RESEARCH
echo =====================================================================
echo.

:: 1. Verify Python Virtual Environment
if not exist "legal_ai_env\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found at .\legal_ai_env
    echo Please make sure the Python virtual environment is set up.
    pause
    exit /b 1
)

:: 2. Check & Start Qdrant Vector Database
echo [1/3] Checking Qdrant Vector Engine...
curl -s http://127.0.0.1:6333/healthz >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo       [OK] Qdrant is already running on port 6333.
) else (
    if exist "bin\qdrant.exe" (
        echo       Starting Qdrant daemon on port 6333...
        start "Juris Qdrant DB" /min "bin\qdrant.exe" --config-path "config\config.yaml"
        timeout /t 3 /nobreak >nul
    ) else (
        echo       [WARNING] bin\qdrant.exe not found. Proceeding with existing server config...
    )
)

:: 3. Check Ollama LLM Service
echo [2/3] Checking Ollama Local LLM Service...
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo       [OK] Ollama is active on port 11434.
) else (
    echo       [INFO] Starting Ollama service...
    start "" /min ollama serve >nul 2>&1
    timeout /t 2 /nobreak >nul
)

:: 4. Launch Web Browser in 3 seconds
echo [3/3] Initializing Juris FastAPI Application...
start "" cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:9010"

:: 5. Run FastAPI Server
echo.
echo =====================================================================
echo   Juris is starting on: http://localhost:9010
echo   Vector DB Dashboard:  http://localhost:6333/dashboard
echo   Admin Portal:         http://localhost:9010/admin
echo =====================================================================
echo   Press CTRL+C in this console to stop the Juris server.
echo =====================================================================
echo.

legal_ai_env\Scripts\python.exe server.py
pause
