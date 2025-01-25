from dataclasses import dataclass
from ten_ai_base.config import BaseConfig


@dataclass
class FunASRConfig(BaseConfig):
    host: str = "127.0.0.1"
    port: int = 10095
    is_ssl: bool = True
    mode: str = "2pass"
    chunk_size: str = "0,10,5"
    chunk_interval: int = 10
    sample_rate: int = 16000
    channels: int = 1
