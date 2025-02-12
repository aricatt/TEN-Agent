@echo off
SET CONDA_PATH=.\Miniconda3

REM Activate environment
call conda activate %CONDA_PATH%

REM Remove CPU-only PyTorch
pip uninstall torch torchvision torchaudio -y

REM Install CUDA-enabled PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y

REM Verify CUDA support
python check_cuda.py

pause
