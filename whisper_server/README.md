# Whisper STT Server

这是一个独立的 Whisper 语音识别服务器，为 TEN-Agent 项目提供语音转文本功能。

## 功能特点

- 基于 OpenAI 的 Whisper 模型
- 支持 CUDA 加速
- RESTful API 接口
- 支持多种语言的语音识别
- 健康检查接口

## 安装

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 确保系统已安装 CUDA 和 cuDNN（如果要使用 GPU 加速）

## 运行

直接运行 `start_server.bat` 或执行：
```bash
python server.py
```

默认服务器会在 8000 端口启动。可以通过环境变量 `WHISPER_SERVER_PORT` 修改端口号。

## API 接口

### 1. 语音识别
- 端点：`POST /transcribe`
- 参数：
  - `file`: 音频文件
  - `model_size`: 模型大小 (tiny/base/small/medium/large)
  - `language`: 目标语言 (如 "zh" 表示中文)
  - `device`: 运行设备 (cuda/cpu)

### 2. 健康检查
- 端点：`GET /health`
- 返回服务器状态和模型加载情况

## 注意事项

1. 服务器直接使用主机的 CUDA 资源，避免了 Docker 环境中的兼容性问题
2. 首次运行时会自动下载指定的 Whisper 模型
3. 建议使用 16kHz 采样率的音频以获得最佳效果
