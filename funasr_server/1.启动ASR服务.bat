@echo off
chcp 65001

SET CONDA_PATH=.\Miniconda3

REM Activate conda environment
call conda activate %CONDA_PATH%

REM Set environment variables for better download experience in China
SET KMP_DUPLICATE_LIB_OK=TRUE
SET CONDA_PATH=.\Miniconda3
set HF_ENDPOINT=https://hf-mirror.com
set MODELSCOPE_CACHE=%CD%\hf_download
set HF_HOME=%CD%\hf_download

REM Add proxy settings for modelscope
set MODELSCOPE_DOMAIN=modelscope.cn
set MODELSCOPE_PROXY=https://modelscope.cn/api/v1

REM Disable auto updates
set disable_update=True

python funasr_wss_server.py --port 10096 --certfile "" --asr_model iic/SenseVoiceSmall --asr_model_revision master --asr_model_online iic/SenseVoiceSmall --asr_model_online_revision master

cmd /k