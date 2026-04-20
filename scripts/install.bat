@echo off
setlocal
cd /d "%~dp0.."

python --version >nul 2>&1
if errorlevel 1 (
    echo Python 3.10+ not found.
    exit /b 1
)

python -m pip install --upgrade pip
python -m pip install -e .

call scripts\download_models.bat
