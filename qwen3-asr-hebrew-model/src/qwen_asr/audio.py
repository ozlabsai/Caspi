"""Audio processing utilities for Qwen3-ASR."""

import re
import tempfile
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf


def normalize_hebrew_text(text: str) -> str:
    """
    Normalize Hebrew text for WER calculation.

    Args:
        text: Hebrew text to normalize

    Returns:
        Normalized text without niqqud and extra whitespace
    """
    # Remove niqqud (Hebrew diacritics)
    text = re.sub(r'[\u0591-\u05C7]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text.strip()


class AudioProcessor:
    """Handles audio loading and conversion from various formats."""

    @staticmethod
    def decode_audio(audio_data) -> tuple[np.ndarray, int]:
        """
        Decode audio from various formats to numpy array.

        Handles:
        - torchcodec AudioDecoder objects
        - Dict with 'array' and 'sampling_rate' keys
        - Tuples/lists of (array, sample_rate)
        - Objects with array and sampling_rate attributes
        - Raw numpy arrays

        Args:
            audio_data: Audio data in any supported format

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        # Handle torchcodec AudioDecoder
        if hasattr(audio_data, 'decode'):
            decoded = audio_data.decode()
            audio_tensor = decoded['audio']
            # Shape is usually (channels, samples), take first channel
            if audio_tensor.ndim == 2:
                audio_array = audio_tensor[0].numpy()
            else:
                audio_array = audio_tensor.numpy()
            sr = int(decoded['sample_rate'])

        # Handle dict format
        elif isinstance(audio_data, dict):
            audio_array = audio_data.get('array')
            sr = audio_data.get('sampling_rate', 16000)

        # Handle tuple/list format
        elif isinstance(audio_data, (list, tuple)) and len(audio_data) >= 2:
            audio_array = audio_data[0]
            sr = audio_data[1]

        # Handle object with attributes
        elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
            audio_array = audio_data.array
            sr = audio_data.sampling_rate

        # Assume raw array
        else:
            audio_array = audio_data
            sr = 16000

        if audio_array is None:
            raise ValueError("Audio array is None")

        return audio_array, sr

    @staticmethod
    def ensure_mono(audio_array: np.ndarray) -> np.ndarray:
        """
        Ensure audio is mono (1D).

        Args:
            audio_array: Audio array (may be multi-channel)

        Returns:
            1D mono audio array
        """
        if not isinstance(audio_array, np.ndarray):
            audio_array = np.array(audio_array)

        if audio_array.ndim > 1:
            # If stereo or multi-channel, take first channel or flatten
            if audio_array.shape[0] > audio_array.shape[1]:
                audio_array = audio_array.flatten()
            else:
                audio_array = audio_array[0]

        # Ensure it's 1D
        return audio_array.squeeze()

    @classmethod
    def save_to_wav(cls, audio_data, output_path: Union[str, Path] = None) -> Path:
        """
        Save audio data to WAV file.

        Args:
            audio_data: Audio in any supported format
            output_path: Output path (creates temp file if None)

        Returns:
            Path to saved WAV file
        """
        audio_array, sr = cls.decode_audio(audio_data)
        audio_array = cls.ensure_mono(audio_array)

        if output_path is None:
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            output_path = tmp.name
            tmp.close()

        sf.write(str(output_path), audio_array, sr)
        return Path(output_path)
