#!/usr/bin/env python3
"""
Qwen3-ASR engine for ivrit-ai evaluation framework.

Usage:
    python evaluate_model.py --engine engines/qwen_asr_engine.py --model OzLabs/Qwen3-ASR-Hebrew-1.7B ...
"""

import io
import time

import librosa
import numpy as np
import torch


def create_app(model_path: str, device: str = "auto"):
    """
    Create a transcription function for Qwen3-ASR model.

    Args:
        model_path: HuggingFace model ID or local path
        device: Device to use ("auto", "cuda:0", "cuda:1", "cpu")

    Returns:
        transcribe_fn(entry) -> (text, transcription_time)
    """
    from qwen_asr import Qwen3ASRModel

    # Determine device
    if device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"Loading Qwen3-ASR model: {model_path}")
    print(f"Device: {device}")

    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
        max_inference_batch_size=32,
        max_new_tokens=4096,
    )

    print("Model loaded successfully")

    def decode_audio(audio_data, target_sr: int = 16000):
        """Decode audio from HuggingFace dataset format."""
        if isinstance(audio_data, dict):
            if 'bytes' in audio_data and audio_data['bytes'] is not None:
                return librosa.load(io.BytesIO(audio_data['bytes']), sr=target_sr)
            elif 'path' in audio_data and audio_data['path'] is not None:
                return librosa.load(audio_data['path'], sr=target_sr)
            elif 'array' in audio_data and audio_data['array'] is not None:
                arr = np.array(audio_data['array'])
                sr = audio_data.get('sampling_rate', target_sr)
                if sr != target_sr:
                    arr = librosa.resample(arr, orig_sr=sr, target_sr=target_sr)
                return arr, target_sr
        raise ValueError(f"Cannot decode audio from: {type(audio_data)}")

    def transcribe_with_chunking(audio_array, sr, chunk_duration: float = 30.0):
        """Transcribe audio with chunking for long files."""
        duration = len(audio_array) / sr

        if duration <= chunk_duration:
            results = model.transcribe(audio=(audio_array, sr), language=None)
            return results[0].text if results else ""

        # Chunk the audio
        chunk_samples = int(chunk_duration * sr)
        transcriptions = []

        for start in range(0, len(audio_array), chunk_samples):
            end = min(start + chunk_samples, len(audio_array))
            chunk = audio_array[start:end]
            results = model.transcribe(audio=(chunk, sr), language=None)
            if results and results[0].text:
                transcriptions.append(results[0].text)

        return " ".join(transcriptions)

    def transcribe_fn(entry):
        """
        Transcribe an entry from the dataset.

        Args:
            entry: Dataset entry with 'audio' field

        Returns:
            (transcription_text, transcription_time_seconds, audio_duration_seconds)
        """
        audio_data = entry["audio"]

        # Check if already decoded (array present)
        if isinstance(audio_data, dict) and 'array' in audio_data and audio_data['array'] is not None:
            audio_array = np.array(audio_data['array'])
            sr = audio_data.get('sampling_rate', 16000)
            if sr != 16000:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
                sr = 16000
        else:
            audio_array, sr = decode_audio(audio_data)

        # Ensure mono
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=0)

        # Calculate audio duration
        audio_duration = len(audio_array) / sr

        start_time = time.time()
        text = transcribe_with_chunking(audio_array, sr)
        elapsed = time.time() - start_time

        return text, elapsed, audio_duration

    return transcribe_fn
