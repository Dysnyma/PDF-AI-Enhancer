@echo off
REM ============================================================
REM 扫描版 PDF AI 画质增强工具 — 主入口
REM 用法:
REM   run.bat input\book.pdf
REM   run.bat                          (处理 input\ 下所有 PDF)
REM   run.bat --scale 4 --quality 90
REM   run.bat --no-ocr --model none    (跳过超分，仅 DocRes 修复)
REM   run.bat --device cpu
REM ============================================================
setlocal
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%scripts\setenv.bat"
cd /d "%SCRIPT_DIR%"

if not exist "env\Scripts\python.exe" (
    echo [ERROR] env\ not found. Run scripts\install.bat first.
    exit /b 1
)

"env\Scripts\python.exe" -m src.main %*
endlocal
