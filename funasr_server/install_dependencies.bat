@echo off
SET CONDA_PATH=.\Miniconda3

REM Activate environment
call conda activate %CONDA_PATH%

REM Install ffmpeg
conda install ffmpeg -y

echo Dependencies installed successfully!
pause
