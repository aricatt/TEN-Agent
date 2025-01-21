from dataclasses import dataclass
from typing import ClassVar, Dict, List
import time
from ten import AsyncTenEnv
from ten_ai_base.config import BaseConfig

@dataclass
class WhisperConfig(BaseConfig):
    """Configuration class for Whisper ASR extension.
    
    Attributes:
        model_size: Size of the Whisper model (tiny, base, small, medium, large)
        device: Device to run inference on (cpu only for stability)
        compute_type: Computation type (int8 for CPU)
        model_path: Optional path to custom model
        language: Language code for transcription
        sample_rate: Audio sample rate in Hz
        beam_size: Beam size for beam search
        log_level: Logging verbosity level
        server_url: URL of the server
    """
    
    # Model settings
    model_size: str = "medium"
    device: str = "cpu"  # CPU-only for stability
    compute_type: str = "int8"  # int8 for CPU
    model_path: str = ""
    
    # Server settings
    server_url: str = "http://localhost:8000"
    
    # Transcription settings
    language: str = "zh"
    sample_rate: int = 16000
    beam_size: int = 5
    log_level: str = "info"

    # Property mappings
    STRING_PROPS: ClassVar[List[str]] = [
        'model_size', 'device', 'compute_type', 
        'model_path', 'language', 'log_level',
        'server_url'
    ]
    INT_PROPS: ClassVar[List[str]] = ['sample_rate', 'beam_size']

    @classmethod
    async def create_async(cls, ten_env: AsyncTenEnv) -> 'WhisperConfig':
        """Create a WhisperConfig instance asynchronously from environment properties.
        
        Args:
            ten_env: AsyncTenEnv instance for property access and logging
            
        Returns:
            WhisperConfig: Configured instance
            
        Raises:
            Exception: If critical properties cannot be loaded
        """
        config = cls()
        try:
            # Load string properties
            for field in cls.STRING_PROPS:
                try:
                    value = await ten_env.get_property_string(field)
                    if value and value.strip():
                        setattr(config, field, value)
                except Exception as e:
                    ten_env.log_warn(f"[whisper_asr_python] Optional property {field} not found: {str(e)}")

            # Load integer properties
            for field in cls.INT_PROPS:
                try:
                    value = await ten_env.get_property_int(field)
                    if value and value > 0:
                        setattr(config, field, value)
                except Exception as e:
                    ten_env.log_warn(f"[whisper_asr_python] Optional property {field} not found: {str(e)}")

            # Force CPU usage for stability
            config.device = "cpu"
            config.compute_type = "int8"
            
            # Log final configuration
            ten_env.log_info("[whisper_asr_python] Configuration loaded successfully:")
            ten_env.log_info(f"[whisper_asr_python] Model: {config.model_size}, Device: {config.device}, Compute: {config.compute_type}")
            ten_env.log_info(f"[whisper_asr_python] Language: {config.language}, Sample Rate: {config.sample_rate}, Beam Size: {config.beam_size}")

            return config
        except Exception as e:
            ten_env.log_warn(f"[whisper_asr_python] Error loading config: {str(e)}")
            raise
