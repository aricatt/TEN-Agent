@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

:: Set environment name and Python version
set "CONDA_ENV=_conda"
set "PYTHON_VERSION=3.11"

:: Check if conda is installed
where conda >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Error: Conda is not installed or not in PATH
    echo Please install Miniconda or Anaconda first
    pause
    exit /b 1
)

:: Check if environment exists
conda env list | findstr /C:"%CONDA_ENV%" >nul
if %ERRORLEVEL% neq 0 (
    echo Creating new conda environment: %CONDA_ENV%
    conda create -y -n %CONDA_ENV% python=%PYTHON_VERSION%
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to create conda environment
        pause
        exit /b 1
    )
) else (
    echo Conda environment %CONDA_ENV% already exists
)

:: Activate environment
echo Activating conda environment: %CONDA_ENV%
call conda activate %CONDA_ENV%
if !ERRORLEVEL! neq 0 (
    echo Error: Failed to activate conda environment
    pause
    exit /b 1
)

:: Install/Update PyTorch with CUDA support
echo Installing/Updating PyTorch with CUDA support...
pip install torch --index-url https://download.pytorch.org/whl/cu118
if !ERRORLEVEL! neq 0 (
    echo Warning: Failed to install PyTorch with CUDA support
    echo Falling back to CPU version
    pip install torch
)

:: Install other dependencies
echo Installing/Updating other dependencies...
pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo Error: Failed to install dependencies
    pause
    exit /b 1
)

:: Check if config file exists
if not exist config.env (
    echo Warning: config.env not found, creating from template...
    (
        echo # Whisper Model Configuration
        echo WHISPER_MODEL=medium     # Options: tiny, base, small, medium, large
        echo WHISPER_DEVICE=cuda     # Options: cuda, cpu
        echo WHISPER_LANGUAGE=zh     # Default language
        echo.
        echo # Server Configuration
        echo SERVER_PORT=8000        # Server port
        echo LOG_LEVEL=INFO         # Log level
    ) > config.env
)

:: Start server
echo Starting Whisper STT server...
python server.py

:: If server exits with error
if !ERRORLEVEL! neq 0 (
    echo Server stopped with error code: !ERRORLEVEL!
    pause
)
