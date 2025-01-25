from ten import (
    AsyncExtension,
    AsyncTenEnv,
    Cmd,
    Data,
    AudioFrame,
    StatusCode,
    CmdResult,
)

import asyncio
import json
import ssl
import websockets
import time

from .config import FunASRConfig

DATA_OUT_TEXT_DATA_PROPERTY_TEXT = "text"
DATA_OUT_TEXT_DATA_PROPERTY_IS_FINAL = "is_final"
DATA_OUT_TEXT_DATA_PROPERTY_STREAM_ID = "stream_id"
DATA_OUT_TEXT_DATA_PROPERTY_END_OF_SEGMENT = "end_of_segment"


class FunASRExtension(AsyncExtension):
    def __init__(self, name: str):
        super().__init__(name)

        self.stopped = False
        self.connected = False
        self.websocket = None
        self.config: FunASRConfig = None
        self.ten_env: AsyncTenEnv = None
        self.loop = None
        self.stream_id = -1
        self.audio_buffer = bytearray()  # 音频缓冲区
        self.chunk_size = None  # 将在配置加载后计算
        self.text_buffer = {}  # 格式: {stream_id: {"online": "", "offline": ""}}
        self.last_text_time = {}  # 格式: {stream_id: timestamp}
        self.text_buffer_time = 2.0  # 实时文本的缓冲时间（秒）
        self.min_text_interval = 0.5  # 最小文本发送间隔（秒）

    async def on_init(self, ten_env: AsyncTenEnv) -> None:
        self.ten_env = ten_env
        ten_env.log_info("FunASRExtension on_init")

    async def on_start(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_info("on_start")
        self.loop = asyncio.get_event_loop()
        self.ten_env = ten_env

        self.config = await FunASRConfig.create_async(ten_env=ten_env)
        ten_env.log_info(f"config: {self.config}")
        
        # 计算每个chunk的大小（以字节为单位）
        chunk_ms = 60 * int(self.config.chunk_size.split(',')[1]) / self.config.chunk_interval  # 毫秒
        samples = int(self.config.sample_rate * chunk_ms / 1000)  # 采样点数
        self.chunk_size = samples * 2  # 16位采样，每个采样2字节
        ten_env.log_info(f"Calculated chunk size: {self.chunk_size} bytes ({samples} samples)")

        self.loop.create_task(self._start_listen())

        ten_env.log_info("starting funasr_wrapper thread")

    async def on_audio_frame(self, _: AsyncTenEnv, frame: AudioFrame) -> None:
        frame_buf = frame.get_buf()

        if not frame_buf:
            self.ten_env.log_warn("send_frame: empty pcm_frame detected.")
            return

        if not self.connected:
            return

        self.stream_id = frame.get_property_int("stream_id")
        
        # 将新的音频数据添加到缓冲区
        self.audio_buffer.extend(frame_buf)
        
        # 当缓冲区达到chunk大小时发送数据
        while len(self.audio_buffer) >= self.chunk_size:
            chunk = self.audio_buffer[:self.chunk_size]
            self.audio_buffer = self.audio_buffer[self.chunk_size:]
            
            if self.websocket:
                await self.websocket.send(chunk)
                
                # 添加一个与chunk_interval匹配的延迟
                sleep_duration = self.config.chunk_interval / 1000  # 转换为秒
                await asyncio.sleep(sleep_duration)

    async def on_stop(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_info("on_stop")

        self.stopped = True

        if self.websocket:
            message = json.dumps({"is_speaking": False})
            await self.websocket.send(message)
            await self.websocket.close()

    async def on_cmd(self, ten_env: AsyncTenEnv, cmd: Cmd) -> None:
        cmd_json = cmd.to_json()
        ten_env.log_info(f"on_cmd json: {cmd_json}")

        cmd_result = CmdResult.create(StatusCode.OK)
        cmd_result.set_property_string("detail", "success")
        await ten_env.return_result(cmd_result, cmd)

    async def _start_listen(self) -> None:
        self.ten_env.log_info("start and listen funasr")

        try:
            if self.config.is_ssl:
                ssl_context = ssl.SSLContext()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
                uri = f"wss://{self.config.host}:{self.config.port}"
            else:
                uri = f"ws://{self.config.host}:{self.config.port}"
                ssl_context = None

            self.ten_env.log_info(f"Connecting to {uri}")
            async with websockets.connect(uri, ssl=ssl_context) as websocket:
                self.websocket = websocket
                self.connected = True

                # Send initial configuration message
                chunk_size = [int(x) for x in self.config.chunk_size.split(",")]
                message = json.dumps({
                    "mode": self.config.mode,
                    "chunk_size": chunk_size,
                    "chunk_interval": self.config.chunk_interval,
                    "encoder_chunk_look_back": 4,
                    "decoder_chunk_look_back": 1,
                    "wav_name": "stream",
                    "is_speaking": True,
                    "audio_fs": self.config.sample_rate,
                    "wav_format": "pcm",
                    "hotwords": "",
                    "itn": True
                })
                await self.websocket.send(message)

                async for message in self.websocket:
                    try:
                        msg = json.loads(message)
                    except json.JSONDecodeError as e:
                        self.ten_env.log_error(f"Failed to parse message as JSON: {e}")
                        continue
                    
                    # Process the message
                    await self._process_message(msg)

        except Exception as e:
            self.ten_env.log_error(f"Failed to connect to FunASR: {str(e)}")
            self.connected = False
            if not self.stopped:
                self.ten_env.log_warn(
                    "FunASR connection failed. Retrying in 1 second..."
                )
                await asyncio.sleep(1)
                self.loop.create_task(self._start_listen())

    async def _process_message(self, msg):
        if not msg or "text" not in msg:
            return

        text = msg["text"]
        mode = msg.get("mode", "")  # 获取消息模式
        is_final = msg.get("is_final", False)

        # 清理特殊格式标记
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]  # 移除首尾的方括号
        
        # 移除所有特殊标记，如<|en|>、<|EMO_UNKNOWN|>等
        import re
        text = re.sub(r'<\|[^|]+\|>', '', text)
        
        self.ten_env.log_info(
            f"funasr got text: [{text}], mode: {mode}, is_final: {is_final}, stream_id: {self.stream_id}"
        )

        # 根据2pass模式处理
        if mode == "2pass-online":
            # 实时结果，暂存在缓冲区
            if self.stream_id not in self.text_buffer:
                self.text_buffer[self.stream_id] = {"online": "", "offline": ""}
            self.text_buffer[self.stream_id]["online"] = text
            
            # 只在达到缓冲时间时发送实时结果
            current_time = time.time()
            if self.stream_id in self.last_text_time:
                time_since_last = current_time - self.last_text_time[self.stream_id]
                if time_since_last >= self.min_text_interval:
                    await self._send_text(text=text, is_final=False, stream_id=self.stream_id)
                    self.last_text_time[self.stream_id] = current_time
            else:
                self.last_text_time[self.stream_id] = current_time
                
        elif mode == "2pass-offline":
            # 离线优化结果，这是一个完整的识别结果
            if self.stream_id in self.text_buffer:
                self.text_buffer[self.stream_id]["offline"] = text
                # 清除在线缓冲
                self.text_buffer[self.stream_id]["online"] = ""
                # 发送最终结果
                await self._send_text(text=text, is_final=True, stream_id=self.stream_id)
                # 清理缓冲区
                del self.text_buffer[self.stream_id]
                if self.stream_id in self.last_text_time:
                    del self.last_text_time[self.stream_id]

    async def _send_text(self, text: str, is_final: bool, stream_id: int) -> None:
        # 只有当文本非空时才发送
        if not text.strip():
            return
            
        stable_data = Data.create("text_data")
        stable_data.set_property_bool(DATA_OUT_TEXT_DATA_PROPERTY_IS_FINAL, is_final)
        stable_data.set_property_string(DATA_OUT_TEXT_DATA_PROPERTY_TEXT, text)
        stable_data.set_property_int(DATA_OUT_TEXT_DATA_PROPERTY_STREAM_ID, stream_id)
        stable_data.set_property_bool(
            DATA_OUT_TEXT_DATA_PROPERTY_END_OF_SEGMENT, is_final
        )
        
        await self.ten_env.send_data(stable_data)
