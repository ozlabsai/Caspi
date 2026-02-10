"""Audio processing utilities for Qwen3-ASR."""

import io
import re
import tempfile
from pathlib import Path
from typing import Union

import librosa
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
    def decode_audio(audio_data, target_sr: int = 16000) -> tuple[np.ndarray, int]:
        """
        Decode audio from various formats to numpy array.

        Handles:
        - Dict with 'bytes' key (raw audio bytes - decoded with librosa)
        - Dict with 'array' and 'sampling_rate' keys (already decoded)
        - Dict with 'path' key (file path)
        - torchcodec AudioDecoder objects
        - Tuples/lists of (array, sample_rate)
        - Objects with array and sampling_rate attributes
        - Raw bytes (decoded with librosa)
        - Raw numpy arrays

        Args:
            audio_data: Audio data in any supported format
            target_sr: Target sample rate for resampling (default 16000)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        audio_array = None
        sr = target_sr

        # Handle dict format (most common from HuggingFace datasets)
        if isinstance(audio_data, dict):
            # Check for raw bytes (needs decoding with librosa)
            if 'bytes' in audio_data and audio_data['bytes'] is not None:
                audio_bytes = audio_data['bytes']
                audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=target_sr)
            # Check for file path
            elif 'path' in audio_data and audio_data['path'] is not None:
                audio_array, sr = librosa.load(audio_data['path'], sr=target_sr)
            # Check for pre-decoded array
            elif 'array' in audio_data and audio_data['array'] is not None:
                audio_array = audio_data['array']
                sr = audio_data.get('sampling_rate', target_sr)
                # Resample if needed
                if sr != target_sr:
                    audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr

        # Handle raw bytes
        elif isinstance(audio_data, bytes):
            audio_array, sr = librosa.load(io.BytesIO(audio_data), sr=target_sr)

        # Handle torchcodec AudioDecoder
        elif hasattr(audio_data, 'decode'):
            decoded = audio_data.decode()
            audio_tensor = decoded['audio']
            # Shape is usually (channels, samples), take first channel
            if audio_tensor.ndim == 2:
                audio_array = audio_tensor[0].numpy()
            else:
                audio_array = audio_tensor.numpy()
            sr = int(decoded['sample_rate'])
            # Resample if needed
            if sr != target_sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

        # Handle tuple/list format
        elif isinstance(audio_data, (list, tuple)) and len(audio_data) >= 2:
            audio_array = audio_data[0]
            sr = audio_data[1]
            if sr != target_sr:
                audio_array = librosa.resample(np.array(audio_array), orig_sr=sr, target_sr=target_sr)
                sr = target_sr

        # Handle object with attributes
        elif hasattr(audio_data, 'array') and hasattr(audio_data, 'sampling_rate'):
            audio_array = audio_data.array
            sr = audio_data.sampling_rate
            if sr != target_sr:
                audio_array = librosa.resample(np.array(audio_array), orig_sr=sr, target_sr=target_sr)
                sr = target_sr

        # Assume raw numpy array
        elif isinstance(audio_data, np.ndarray):
            audio_array = audio_data
            sr = target_sr

        if audio_array is None:
            raise ValueError(f"Could not decode audio from type: {type(audio_data)}")

        return np.array(audio_array), sr

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
