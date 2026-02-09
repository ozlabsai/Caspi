#!/usr/bin/env python3
"""
Production FastAPI server for Qwen3-ASR Hebrew model.

Deploy to any GPU instance (Lambda Labs, RunPod, AWS, etc.)

Usage:
    uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000
"""

import io
import base64
import logging
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import soundfile as sf

# Import Qwen3-ASR
try:
    from qwen_asr import Qwen3ASRModel
except ImportError:
    raise ImportError(
        "qwen-asr package not found. Install with: pip install qwen-asr"
    )

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Qwen3-ASR Hebrew API",
    description="Hebrew speech-to-text using fine-tuned Qwen3-ASR",
    version="1.0.0"
)

# Global model variable (loaded on startup)
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = Path(__file__).parent  # Current directory


class TranscribeRequest(BaseModel):
    """Request body for base64-encoded audio."""
    audio_base64: str
    language: str = "Hebrew"
    sample_rate: int = 16000


class TranscribeResponse(BaseModel):
    """Response with transcription."""
    text: str
    language: str
    model: str


@app.on_event("startup")
async def load_model():
    """Load model on server startup."""
    global MODEL

    logger.info(f"Loading Qwen3-ASR model from {MODEL_PATH}")
    logger.info(f"Using device: {DEVICE}")

    try:
        MODEL = Qwen3ASRModel.from_pretrained(
            str(MODEL_PATH),
            device_map=DEVICE,
            torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
        )
        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "model": "qwen3-asr-hebrew",
        "device": DEVICE,
        "version": "1.0.0"
    }


@app.get("/health")
async def health():
    """Detailed health check."""
    return {
        "status": "healthy" if MODEL is not None else "unhealthy",
        "model_loaded": MODEL is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    }


@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_base64(request: TranscribeRequest):
    """
    Transcribe audio from base64-encoded string.

    Expects audio as base64-encoded WAV/FLAC/MP3.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Decode base64 audio
        audio_bytes = base64.b64decode(request.audio_base64)

        # Load audio using soundfile
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample if needed (Qwen3-ASR expects 16kHz)
        if sr != 16000:
            logger.warning(f"Resampling from {sr}Hz to 16000Hz")
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sr, target_sr=16000
            )

        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Transcribe
        logger.info(f"Transcribing {len(audio_array)/16000:.2f}s audio")
        result = MODEL.transcribe(
            audio_array,
            language=request.language
        )

        # Extract text
        text = result['text'] if isinstance(result, dict) else result

        return TranscribeResponse(
            text=text,
            language=request.language,
            model="qwen3-asr-hebrew"
        )

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe/file")
async def transcribe_file(
    file: UploadFile = File(...),
    language: str = "Hebrew"
):
    """
    Transcribe audio from uploaded file.

    Accepts WAV, FLAC, MP3, OGG formats.
    """
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read uploaded file
        audio_bytes = await file.read()

        # Load audio
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))

        # Resample if needed
        if sr != 16000:
            logger.warning(f"Resampling from {sr}Hz to 16000Hz")
            import librosa
            audio_array = librosa.resample(
                audio_array, orig_sr=sr, target_sr=16000
            )

        # Ensure mono
        if len(audio_array.shape) > 1:
            audio_array = audio_array.mean(axis=1)

        # Transcribe
        logger.info(f"Transcribing {file.filename} ({len(audio_array)/16000:.2f}s)")
        result = MODEL.transcribe(
            audio_array,
            language=language
        )

        # Extract text
        text = result['text'] if isinstance(result, dict) else result

        return {
            "text": text,
            "language": language,
            "model": "qwen3-asr-hebrew",
            "filename": file.filename,
            "duration_seconds": len(audio_array) / 16000
        }

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    # Run with: python serve_asr.py
    # Or better: uv run fastapi serve_asr.py --host 0.0.0.0 --port 8000
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
