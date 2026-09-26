@echo off
REM ============================================================
REM 一次性安装：项目独立 venv + 全部依赖（约 3.5GB，全部在项目内）
REM 不需要管理员权限，不修改系统环境
REM ============================================================
setlocal
set "SCRIPT_DIR=%~dp0"
call "%SCRIPT_DIR%setenv.bat"

REM 自动检测基础 Python（按优先级：py launcher -> python -> 常见安装路径）
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo [ERROR] Python not found. Install Python 3.10+ and ensure it is on PATH.
    exit /b 1
)

if exist "%SCRIPT_DIR%..\env\Scripts\python.exe" (
    echo [INFO] env\ already exists, upgrading deps only
) else (
    echo [INFO] creating venv in env\ ...
    %PY% -m venv "%SCRIPT_DIR%..\env"
)

set "VENV_PY=%SCRIPT_DIR%..\env\Scripts\python.exe"

echo [INFO] installing PyTorch cu126 ...
"%VENV_PY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
echo [INFO] installing remaining deps ...
"%VENV_PY%" -m pip install pymupdf opencv-python-headless pillow numpy pyyaml einops

echo [INFO] done. Use run.bat to process PDFs.
endlocal
