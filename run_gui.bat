@echo off
REM ============================================================
REM Scanned PDF AI Enhancer - GUI launcher
REM Double-click to start the local web GUI.
REM The browser opens automatically once the server is ready.
REM ============================================================
setlocal
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%scripts\setenv.bat"
cd /d "%SCRIPT_DIR%"

if not exist "env\Scripts\python.exe" (
    echo [ERROR] env\ not found. Run scripts\install.bat first.
    pause
    exit /b 1
)

set "PORT=8765"
if not "%~1"=="" set "PORT=%~1"

echo.
echo   Scanned PDF AI Enhancer - GUI
echo   ---------------------------------------------
echo   Starting server on port %PORT% ...
echo   The browser will open automatically.
echo   Press Ctrl+C here to stop the server.
echo.

"env\Scripts\python.exe" -m src.gui %PORT%
pause
