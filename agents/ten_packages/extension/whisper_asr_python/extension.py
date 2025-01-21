from ten import (
    AsyncExtension,
    AsyncTenEnv,
    Cmd,
    Data,
    AudioFrame,
    StatusCode,
    CmdResult,
)
import os
import json
import asyncio
import numpy as np
import requests
import traceback
import logging
from typing import Optional
from ten_ai_base.config import BaseConfig
from .config import WhisperConfig
from faster_whisper import WhisperModel
import time
from dataclasses import dataclass

DATA_OUT_TEXT_DATA_PROPERTY_TEXT = "text"
DATA_OUT_TEXT_DATA_PROPERTY_IS_FINAL = "is_final"
DATA_OUT_TEXT_DATA_PROPERTY_STREAM_ID = "stream_id"
DATA_OUT_TEXT_DATA_PROPERTY_END_OF_SEGMENT = "end_of_segment"


@dataclass
class WhisperConfig(BaseConfig):
    """Whisper配置类"""
    model_size: str = "medium"  # 模型大小
    device: str = "cuda"  # 设备类型
    compute_type: str = "float16"  # 计算类型
    model_path: str = ""  # 模型路径
    language: str = "zh"  # 语言
    sample_rate: int = 16000  # 采样率
    beam_size: int = 5  # 束搜索大小
    log_level: str = "info"  # 日志级别
    server_url: str = "http://localhost:8000"  # Whisper 服务器地址
    STRING_PROPS = ["model_size", "device", "compute_type", "model_path", "language", "log_level", "server_url"]

    async def load_from_env(self, ten_env):
        """从环境中加载配置"""
        try:
            # 加载字符串属性
            for prop in self.STRING_PROPS:
                try:
                    val = await ten_env.get_property_string(prop)
                    if val:
                        setattr(self, prop, val)
                except Exception as e:
                    ten_env.log_warn(f"[whisper_asr_python] Failed to load {prop}: {e}")

            # 加载数值属性
            try:
                sample_rate = await ten_env.get_property_int("sample_rate")
                if sample_rate:
                    self.sample_rate = sample_rate
            except Exception as e:
                ten_env.log_warn(f"[whisper_asr_python] Failed to load sample_rate: {e}")

            try:
                beam_size = await ten_env.get_property_int("beam_size")
                if beam_size:
                    self.beam_size = beam_size
            except Exception as e:
                ten_env.log_warn(f"[whisper_asr_python] Failed to load beam_size: {e}")

            ten_env.log_info(f"[whisper_asr_python] Loaded config: {self}")
        except Exception as e:
            ten_env.log_error(f"[whisper_asr_python] Error loading configuration: {e}")
            raise

    def __str__(self):
        """返回配置的字符串表示"""
        return (f"WhisperConfig(model_size='{self.model_size}', device='{self.device}', "
                f"compute_type='{self.compute_type}', model_path='{self.model_path}', "
                f"language='{self.language}', sample_rate={self.sample_rate}, "
                f"beam_size={self.beam_size}, log_level='{self.log_level}', server_url='{self.server_url}')")


class WhisperASR:
    def __init__(self, config: WhisperConfig):
        """Initialize WhisperASR with config."""
        self.config = config
        self.audio_buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 0
        self.target_duration = 2.0  # seconds
        self.session = requests.Session()
        self.min_audio_length = 1.0  # 最小音频长度（秒）
        self.request_id = 0  # 请求计数器
        
        # 音频质量参数
        self.noise_threshold = 0.005  # 噪音阈值
        self.signal_threshold = 0.02  # 有效信号阈值
        self.min_signal_ratio = 0.1  # 最小信号比例

    def is_valid_audio(self, audio_data: np.ndarray) -> bool:
        """检查音频是否有效（不是噪音）"""
        # 计算音频统计信息
        abs_data = np.abs(audio_data)
        max_amplitude = np.max(abs_data)
        rms = np.sqrt(np.mean(np.square(audio_data)))
        
        # 计算信号部分（超过噪音阈值的样本）
        signal_samples = abs_data > self.noise_threshold
        signal_ratio = np.sum(signal_samples) / len(audio_data)
        
        # 检查是否是有效的语音信号
        if max_amplitude < self.signal_threshold:
            if self._should_log('debug'):
                logging.debug("[whisper_asr_python] Audio signal too weak")
            return False
            
        if signal_ratio < self.min_signal_ratio:
            if self._should_log('debug'):
                logging.debug("[whisper_asr_python] Too few signal samples")
            return False
            
        if rms < self.noise_threshold:
            if self._should_log('debug'):
                logging.debug("[whisper_asr_python] Audio mostly noise")
            return False
            
        return True

    def _should_log(self, level: str) -> bool:
        """检查是否应该输出日志"""
        return logging.getLogger().isEnabledFor(getattr(logging, level.upper()))

    def process_audio(self, audio_data: np.ndarray) -> str:
        """Process audio data and return transcription."""
        try:
            # 检查音频长度
            audio_duration = len(audio_data) / self.config.sample_rate
            if audio_duration < self.min_audio_length:
                if self._should_log('debug'):
                    logging.debug(f"[whisper_asr_python] Audio too short: {audio_duration}s < {self.min_audio_length}s")
                return ""

            # 确保音频数据是 float32 类型，范围在 [-1, 1] 之间
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 1.0:
                audio_data = audio_data / 32768.0
                
            # 检查音频质量
            if not self.is_valid_audio(audio_data):
                return ""

            # 将音频数据转换为 16kHz 采样率的 16 位整数 PCM
            audio_data = (audio_data * 32768.0).astype(np.int16)
            
            # 生成唯一的请求ID
            self.request_id += 1
            
            # 准备上传的文件
            audio_bytes = audio_data.tobytes()
            
            files = {
                'file': ('audio.raw', audio_bytes, 'application/octet-stream')
            }

            # 准备参数
            params = {
                'language': self.config.language,
                'sample_rate': str(self.config.sample_rate),
                'model_size': self.config.model_size,
                'compute_type': self.config.compute_type,
                'beam_size': str(self.config.beam_size),
                'request_id': str(self.request_id)
            }

            # 发送请求
            response = self.session.post(
                f"{self.config.server_url}/transcribe",
                files=files,
                data=params,
                timeout=30,
                headers={'Cache-Control': 'no-cache'}
            )
            response.raise_for_status()
            result = response.json()
            logging.info(f"[whisper_asr_python] Server response (request_id={self.request_id}): {result}")

            # 返回转写结果
            text = result.get("text", "").strip()
            if text:
                logging.info(f"[whisper_asr_python] Transcribed: {text}")
            return text

        except requests.exceptions.RequestException as e:
            logging.error(f"[whisper_asr_python] Network error: {str(e)}")
            return ""
        except Exception as e:
            logging.error(f"[whisper_asr_python] Error: {str(e)}")
            if self._should_log('debug'):
                logging.debug(f"[whisper_asr_python] Traceback: {traceback.format_exc()}")
            return ""


class WhisperASRExtension(AsyncExtension):
    def __init__(self, name: str):
        super().__init__(name)
        self.config = None
        self.asr = None
        self.sample_width = 2  # 16-bit audio
        self.channels = 1  # mono audio
        self.stream_id = 0
        self.ten_env = None

    def _should_log(self, level: str) -> bool:
        """检查是否应该输出日志"""
        return logging.getLogger().isEnabledFor(getattr(logging, level.upper()))

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        """初始化"""
        self.ten_env = ten_env
        self.ten_env.log_info("[whisper_asr_python] on_init")
        self.config = WhisperConfig()
        await self.config.load_from_env(ten_env)

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        """启动"""
        self.ten_env = ten_env
        self.ten_env.log_info("[whisper_asr_python] on_start")
        self.ten_env.log_info(f"[whisper_asr_python] config: {self.config}")

        # 初始化 WhisperASR 实例
        self.asr = WhisperASR(self.config)

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        """停止"""
        self.ten_env = ten_env
        self.ten_env.log_info("[whisper_asr_python] on_stop")

    async def on_deinit(self, ten_env: AsyncTenEnv) -> None:
        """反初始化"""
        self.ten_env = ten_env
        self.ten_env.log_info("[whisper_asr_python] on_deinit")

    async def on_data(self, ten_env: AsyncTenEnv, data: Data) -> None:
        """处理数据"""
        if self._should_log('debug'):
            ten_env.log_debug(f"[whisper_asr_python] Received data: {data.name}")

        if isinstance(data, AudioFrame):
            await self.on_audio_frame(ten_env, data)

    async def on_audio_frame(self, ten_env: AsyncTenEnv, frame: AudioFrame) -> None:
        """处理音频帧"""
        try:
            # 获取音频数据
            audio_data = frame.get_buf()
            if not audio_data:
                if self._should_log('debug'):
                    ten_env.log_debug("[whisper_asr_python] Empty audio frame detected.")
                return

            # 获取流ID
            self.stream_id = frame.get_property_int("stream_id")

            # 将音频数据转换为numpy数组
            audio_array = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

            # 确保 asr 实例已初始化
            if not hasattr(self, 'asr') or self.asr is None:
                self.asr = WhisperASR(self.config)
                ten_env.log_info("[whisper_asr_python] WhisperASR instance initialized")

            # 将音频数据添加到缓冲区
            if not hasattr(self.asr, 'audio_buffer'):
                self.asr.audio_buffer = np.array([], dtype=np.float32)
                self.asr.buffer_duration = 0
                ten_env.log_info("[whisper_asr_python] Audio buffer initialized")

            self.asr.audio_buffer = np.concatenate([self.asr.audio_buffer, audio_array])
            self.asr.buffer_duration = len(self.asr.audio_buffer) / self.config.sample_rate

            # 如果缓冲区中的音频长度达到目标长度，则进行处理
            if self.asr.buffer_duration >= self.asr.target_duration:
                #ten_env.log_info(f"[whisper_asr_python] Processing {self.asr.buffer_duration:.2f}s audio")
                
                # 处理音频并获取转录结果
                audio_to_process = self.asr.audio_buffer.copy()
                self.asr.audio_buffer = np.array([], dtype=np.float32)
                self.asr.buffer_duration = 0
                
                transcription = self.asr.process_audio(audio_to_process)

                # 发送转录结果
                if transcription:
                    ten_env.log_info(f"[whisper_asr_python] STT Result: {transcription}")
                    await self._send_text(ten_env, transcription, True, self.stream_id)
                #else:
                #    ten_env.log_info("[whisper_asr_python] No transcription result")

        except Exception as e:
            ten_env.log_error(f"[whisper_asr_python] Error: {str(e)}")
            if self._should_log('debug'):
                ten_env.log_error(f"[whisper_asr_python] Traceback: {traceback.format_exc()}")

    async def _send_text(self, ten_env: AsyncTenEnv, text: str, is_final: bool, stream_id: int) -> None:
        """发送文本"""
        try:
            # 创建数据包
            data = Data.create("text_data")
            data.set_property_string("text", text)
            data.set_property_bool("is_final", is_final)
            data.set_property_int("stream_id", stream_id)
            data.set_property_bool("end_of_segment", is_final)
            
            ten_env.log_info(f"[whisper_asr_python] Sending text to LLM: {text}")
            
            # 发送数据
            await ten_env.send_data(data)

        except Exception as e:
            ten_env.log_error(f"[whisper_asr_python] Error sending text: {str(e)}")
            if self._should_log('debug'):
                ten_env.log_error(f"[whisper_asr_python] Traceback: {traceback.format_exc()}")
