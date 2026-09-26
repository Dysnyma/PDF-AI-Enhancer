@echo off
REM ============================================================
REM 项目环境隔离 — 所有缓存/临时目录强制指向项目内
REM 由 run.bat 自动调用，无需手动执行或配置系统环境变量
REM ============================================================
set "PROJECT_ROOT=%~dp0.."

set "HF_HOME=%PROJECT_ROOT%\cache\huggingface"
set "TRANSFORMERS_CACHE=%PROJECT_ROOT%\cache\huggingface"
set "HF_HUB_CACHE=%PROJECT_ROOT%\cache\huggingface"
set "TORCH_HOME=%PROJECT_ROOT%\cache\torch"
set "PIP_CACHE_DIR=%PROJECT_ROOT%\cache\pip"
set "XDG_CACHE_HOME=%PROJECT_ROOT%\cache\xdg"
set "GRPC_VERBOSITY=ERROR"

set "TEMP=%PROJECT_ROOT%\temp"
set "TMP=%PROJECT_ROOT%\temp"

REM 阻止 Python 意外使用用户全局 site-packages
set "PYTHONNOUSERSITE=1"
