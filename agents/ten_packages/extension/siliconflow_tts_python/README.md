# siliconflow_tts_python

SiliconFlow TTS extension for TEN Framework, using SiliconFlow's text-to-speech API.

## Features

- High-quality text-to-speech synthesis using SiliconFlow's API
- Support for multiple voices and models
- Adjustable speech parameters (speed, gain)
- Streaming audio output
- MP3 format support with configurable sample rate

## API

Refer to `api` definition in [manifest.json](manifest.json) and default values in [property.json](property.json).

### Configuration

The extension requires the following configuration in property.json:
- `api_key`: Your SiliconFlow API key (required)
- `model`: TTS model name (default: "fishaudio/fish-speech-1.5")
- `voice`: Voice ID (default: "fishaudio/fish-speech-1.5:alex")
- `sample_rate`: Audio sample rate (default: 32000)
- `speed`: Speech speed multiplier (default: 1.0)
- `gain`: Audio gain adjustment (default: 0.0)

## Development

### Dependencies

- Python 3.7+
- requests library

### Build

1. Set up your SiliconFlow API key in the environment:
   ```bash
   export SILICONFLOW_API_KEY=your_api_key_here
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Unit test

Run the unit tests in the tests directory:
```bash
python -m pytest tests/
```
