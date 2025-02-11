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
        self.min_text_interval = 0.1  # 最小文本发送间隔（秒）
        # 添加噪音过滤配置
        self.noise_words = {'啊', '嗯', '呃', 'yah', '啦', '哦', '噢', '嘿', '呀', 'ok'}  # 常见噪音词
        self.min_text_length = 2  # 最小有效文本长度

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
        
        #self.ten_env.log_info(
        #    f"funasr got text: [{text}], mode: {mode}, is_final: {is_final}, stream_id: {self.stream_id}"
        #)

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

                    #self.ten_env.log_info(f"2pass-online text: {text} ")
                    
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

                #self.ten_env.log_info(f"2pass-offline text: {text} ")

                # 发送最终结果
                await self._send_text(text=text, is_final=True, stream_id=self.stream_id)
                # 清理缓冲区
                del self.text_buffer[self.stream_id]
                if self.stream_id in self.last_text_time:
                    del self.last_text_time[self.stream_id]

    def _is_valid_text(self, text: str) -> bool:
        """
        检查文本是否为有效输入（不是噪音）
        """
        # 去除空白字符
        text = text.strip()
        
        # 长度检查
        if len(text) < self.min_text_length:
            #self.ten_env.log_info(f"Text too short, filtered: {text}")
            return False
            
        # 噪音词检查
        if text in self.noise_words:
            #self.ten_env.log_info(f"Noise word filtered: {text}")
            return False
            
        # 纯标点符号检查
        if all(char in '。，？！,.?!' for char in text):
            #self.ten_env.log_info(f"Punctuation only, filtered: {text}")
            return False

        # 英文短语检查
        words = text.split()
        if all(word.replace(',', '').replace('.', '').isascii() and 
               word.replace(',', '').replace('.', '').isalpha() 
               for word in words):
            # 如果全是英文单词，且单词数量在1-3之间，或者是单个字母
            if 1 <= len(words) <= 3 or (len(words) == 1 and len(words[0]) == 1):
                self.ten_env.log_info(f"English phrase or single letter filtered: {text}")
                return False
            
        return True

    def _check_and_remove_name(self, text: str) -> tuple[bool, str]:
        """
        检查文本是否包含 lucy 称呼，如果有则移除
        返回: (是否包含称呼, 处理后的文本)
        """
        # 统一转换为小写进行检查
        text_lower = text.lower()
        name_variations = ["lucy", "露西", "露茜"]
        
        # 检查是否包含任意一个称呼
        for name in name_variations:
            if name in text_lower:
                # 找到实际的称呼（保持原始大小写）
                start_idx = text_lower.find(name)
                end_idx = start_idx + len(name)
                actual_name = text[start_idx:end_idx]
                
                # 移除称呼和可能跟随的标点符号
                text = text.replace(actual_name, "").strip()
                text = text.lstrip(',.，。、 ')
                return True, text
                
        return False, text

    async def _send_text(self, text: str, is_final: bool, stream_id: int) -> None:
        # 检查文本是否包含称呼
        has_name, processed_text = self._check_and_remove_name(text)
        if not has_name:
            #self.ten_env.log_info(f"Text without name prefix filtered: {text}")
            return
            
        # 检查处理后的文本是否有效
        if not self._is_valid_text(processed_text):
            return
            
        #目前funasr生成的文本开头会有错误的标点符号（如：？和。等），需要去掉
        processed_text = processed_text.lstrip('。')
        processed_text = processed_text.lstrip('？')
       
        stable_data = Data.create("text_data")
        stable_data.set_property_bool(DATA_OUT_TEXT_DATA_PROPERTY_IS_FINAL, is_final)
        stable_data.set_property_string(DATA_OUT_TEXT_DATA_PROPERTY_TEXT, processed_text)
        stable_data.set_property_int(DATA_OUT_TEXT_DATA_PROPERTY_STREAM_ID, stream_id)
        stable_data.set_property_bool(
            DATA_OUT_TEXT_DATA_PROPERTY_END_OF_SEGMENT, is_final
        )
        self.ten_env.log_info(f"text: {processed_text}")
        await self.ten_env.send_data(stable_data)
