"""vLLM client wrapper for Qwen3-ASR."""

from pathlib import Path
from typing import Union

from openai import OpenAI


class VLLMClient:
    """Wrapper for vLLM OpenAI-compatible API client."""

    def __init__(self, base_url: str = "http://localhost:8000/v1", api_key: str = "EMPTY"):
        """
        Initialize vLLM client.

        Args:
            base_url: Base URL for vLLM server
            api_key: API key (use "EMPTY" for local vLLM)
        """
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.base_url = base_url

    def transcribe(
        self,
        audio_path: Union[str, Path],
        model: str = "./model",
        language: str = "he"
    ) -> str:
        """
        Transcribe audio file.

        Args:
            audio_path: Path to audio file
            model: Model name/path
            language: Language code

        Returns:
            Transcribed text
        """
        with open(audio_path, "rb") as f:
            result = self.client.audio.transcriptions.create(
                model=model,
                file=f,
                language=language
            )
        return result.text

    def check_health(self) -> bool:
        """
        Check if vLLM server is healthy.

        Returns:
            True if server is healthy
        """
        try:
            models = self.client.models.list()
            return len(models.data) > 0
        except Exception:
            return False

    def get_model_info(self) -> dict:
        """
        Get information about loaded model.

        Returns:
            Dict with model information
        """
        models = self.client.models.list()
        if models.data:
            return {
                "id": models.data[0].id,
                "created": models.data[0].created,
            }
        return {}
