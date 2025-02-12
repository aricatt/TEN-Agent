@echo off

REM Check if conda is installed
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo Conda is not installed. Please install Conda or Miniconda first
    echo You can download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

REM Set Miniconda3 path
SET CONDA_PATH=.\Miniconda3

REM Check if local Miniconda3 directory exists
if not exist %CONDA_PATH% (
    echo Miniconda3 directory does not exist, creating new environment...
    
    REM Create new conda environment in current directory
    conda create --prefix %CONDA_PATH% python=3.8 -y
    
    if %errorlevel% neq 0 (
        echo Failed to create conda environment
        pause
        exit /b 1
    )
    
    echo Conda environment created successfully!
)

REM Activate environment
echo Activating conda environment...
call conda activate %CONDA_PATH%

if %errorlevel% neq 0 (
    echo Failed to activate conda environment
    pause
    exit /b 1
)

REM Install PyTorch with CUDA support and other required packages
echo Installing PyTorch with CUDA support...
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -q funasr modelscope

pip install -r requirements_server.txt

echo Conda environment activated successfully!

cmd /k