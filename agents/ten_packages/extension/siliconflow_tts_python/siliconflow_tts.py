import asyncio
from dataclasses import dataclass
import requests
from ten.async_ten_env import AsyncTenEnv
from ten_ai_base.config import BaseConfig


@dataclass
class SiliconFlowTTSConfig(BaseConfig):
    api_key: str = ""
    voice: str = "FunAudioLLM/CosyVoice2-0.5B:anna"
    model: str = "FunAudioLLM/CosyVoice2-0.5B"
    sample_rate: int = 16000
    speed: float = 1.0
    gain: float = 0.0


class SiliconFlowTTS:
    def __init__(self, config: SiliconFlowTTSConfig) -> None:
        self.config = config
        self.url = "https://api.siliconflow.cn/v1/audio/speech"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }

    def _create_payload(self, text: str):
        return {
            "model": self.config.model,
            "input": text,
            "voice": self.config.voice,
            "response_format": "mp3",
            "sample_rate": self.config.sample_rate,
            "stream": True,
            "speed": self.config.speed,
            "gain": self.config.gain
        }

    def text_to_speech_stream(
        self, ten_env: AsyncTenEnv, text: str, end_of_segment: bool
    ) -> bytes:
        try:
            payload = self._create_payload(text)
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                stream=True
            )
            
            if response.status_code != 200:
                ten_env.log_error(f"API request failed with status code {response.status_code}")
                return None

            return response.content

        except Exception as e:
            ten_env.log_error(f"Error in text_to_speech_stream: {e}")
            return None

    def cancel(self, ten_env: AsyncTenEnv) -> None:
        ten_env.log_info("Cancelling TTS request")
