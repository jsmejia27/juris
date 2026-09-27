@echo off
title Stop Juris
color 0C
cd /d "%~dp0"

echo =====================================================================
echo                  STOPPING JURIS SERVICES
echo =====================================================================
echo.

echo [1/2] Terminating Juris Server (Port 9010)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":9010" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo [2/2] Terminating Qdrant Database (Port 6333)...
for /f "tokens=5" %%a in ('netstat -aon ^| findstr ":6333" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo.
echo [OK] All Juris services have been stopped.
timeout /t 3 >nul
