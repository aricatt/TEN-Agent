import unittest
from unittest.mock import MagicMock, patch
from ..siliconflow_tts import SiliconFlowTTS, SiliconFlowTTSConfig


class TestSiliconFlowTTS(unittest.TestCase):
    def setUp(self):
        self.config = SiliconFlowTTSConfig(
            api_key="test_key",
            voice="test_voice",
            model="test_model",
            sample_rate=32000,
            speed=1.0,
            gain=0.0
        )
        self.tts = SiliconFlowTTS(self.config)
        self.ten_env = MagicMock()

    @patch('requests.post')
    def test_text_to_speech_stream_success(self, mock_post):
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.content = b"test_audio_data"
        mock_post.return_value = mock_response

        result = self.tts.text_to_speech_stream(
            self.ten_env,
            "Test text",
            True
        )

        self.assertEqual(result, b"test_audio_data")
        mock_post.assert_called_once()

    @patch('requests.post')
    def test_text_to_speech_stream_failure(self, mock_post):
        # Mock failed API response
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_post.return_value = mock_response

        result = self.tts.text_to_speech_stream(
            self.ten_env,
            "Test text",
            True
        )

        self.assertIsNone(result)
        self.ten_env.log_error.assert_called_once()


if __name__ == '__main__':
    unittest.main()
