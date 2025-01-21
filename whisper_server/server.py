import os
from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import whisper  # OpenAI's Whisper
import numpy as np
import uvicorn
from typing import Optional
import logging
from dotenv import load_dotenv
import torch  # 添加torch导入

# 检查CUDA可用性
CUDA_AVAILABLE = torch.cuda.is_available()

# 添加详细的CUDA诊断信息
logger = logging.getLogger("whisper_server")
logger.info("=== CUDA Diagnostic Information ===")
logger.info(f"PyTorch version: {torch.__version__}")
logger.info(f"CUDA available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    logger.info(f"CUDA version: {torch.version.cuda}")
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")
    logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
else:
    logger.warning("CUDA is not available. Using CPU instead.")
    logger.info("Please check:")
    logger.info("1. NVIDIA GPU is properly installed")
    logger.info("2. CUDA drivers are installed")
    logger.info("3. PyTorch is installed with CUDA support")

# 加载.env配置文件
load_dotenv("config.env")

# 配置日志
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper()),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# 记录CUDA状态
logger.info(f"CUDA available: {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")

app = FastAPI(title="Whisper STT Server")

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局变量存储 Whisper 模型
model = None

def load_model(model_size: str = None, device: str = None):
    global model
    if model is None:
        # 从环境变量获取配置，如果没有则使用参数值或默认值
        model_size = model_size or os.getenv("WHISPER_MODEL", "medium")
        device = device or os.getenv("WHISPER_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu")
        
        logger.info(f"Loading Whisper model: {model_size} on device: {device}")
        model = whisper.load_model(model_size)
        
        # 只有当设备为cuda且CUDA可用时才移动到GPU
        if device == "cuda" and CUDA_AVAILABLE:
            model = model.cuda()
        else:
            model = model.cpu()
            
        logger.info("Model loaded successfully")
    return model

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile,
    model_size: Optional[str] = None,
    language: Optional[str] = None,
    device: Optional[str] = None
):
    """
    将音频转写为文本
    
    Args:
        file: 音频文件
        model_size: Whisper模型大小 (可选，默认使用环境变量配置)
        language: 目标语言 (可选，默认使用环境变量配置)
        device: 运行设备 (可选，默认使用环境变量配置)
    
    Returns:
        dict: 包含转写文本和其他信息
    """
    try:
        # 加载模型
        model = load_model(model_size, device)
        
        # 读取音频数据
        content = await file.read()
        audio_np = np.frombuffer(content, dtype=np.int16).astype(np.float32) / 32768.0
        
        logger.info(f"Processing audio file: {file.filename}, size: {len(content)} bytes")
        
        # 使用环境变量配置或参数值
        language = language or os.getenv("WHISPER_LANGUAGE", "zh")
        
        # 转写音频
        result = model.transcribe(
            audio_np,
            language=language,
            task="transcribe"
        )
        
        logger.info(f"Transcription completed successfully")
        
        return {
            "text": result["text"],
            "segments": result["segments"],
            "language": result["language"]
        }
        
    except Exception as e:
        logger.error(f"Error during transcription: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": CUDA_AVAILABLE,
        "cuda_device": torch.cuda.get_device_name(0) if CUDA_AVAILABLE else None,
        "config": {
            "model_size": os.getenv("WHISPER_MODEL", "medium"),
            "device": os.getenv("WHISPER_DEVICE", "cuda" if CUDA_AVAILABLE else "cpu"),
            "language": os.getenv("WHISPER_LANGUAGE", "zh"),
            "log_level": os.getenv("LOG_LEVEL", "INFO")
        }
    }

if __name__ == "__main__":
    # 从环境变量获取端口
    port = int(os.getenv("SERVER_PORT", "8000"))
    
    # 启动服务器
    logger.info(f"Starting Whisper STT server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
