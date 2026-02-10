"""Dataset loading utilities for Hebrew ASR evaluation."""

from typing import Optional

from datasets import load_dataset as hf_load_dataset
from datasets import Audio


# Hebrew evaluation datasets configuration
HEBREW_DATASETS = {
    "eval-d1": {
        "name": "ivrit-ai/eval-d1",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "config": None,
    },
    "eval-whatsapp": {
        "name": "ivrit-ai/eval-whatsapp",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "config": None,
    },
    "saspeech": {
        "name": "upai-inc/saspeech",
        "split": "test",
        "audio_col": "audio",
        "text_col": "text",
        "config": None,
    },
    "hebrew-speech-kan": {
        "name": "imvladikon/hebrew_speech_kan",
        "split": "validation",
        "audio_col": "audio",
        "text_col": "sentence",
        "config": None,
    },
}


def load_hebrew_dataset(
    dataset_key: str,
    streaming: bool = True,
    split: Optional[str] = None,
    decode_audio: bool = False
):
    """
    Load a Hebrew ASR dataset.

    Args:
        dataset_key: Key from HEBREW_DATASETS or full dataset name
        streaming: Whether to use streaming mode
        split: Override default split
        decode_audio: If False, returns raw audio bytes (avoids torchcodec).
                     If True, uses default audio decoding (requires torchcodec).

    Returns:
        Dataset iterator

    Raises:
        ValueError: If dataset_key is not found
    """
    if dataset_key in HEBREW_DATASETS:
        config = HEBREW_DATASETS[dataset_key]
        dataset_name = config["name"]
        dataset_split = split or config["split"]
        dataset_config = config["config"]
    else:
        # Assume it's a full dataset name
        dataset_name = dataset_key
        dataset_split = split or "test"
        dataset_config = None

    # Load dataset
    if dataset_config:
        ds = hf_load_dataset(
            dataset_name,
            dataset_config,
            split=dataset_split,
            streaming=streaming
        )
    else:
        ds = hf_load_dataset(
            dataset_name,
            split=dataset_split,
            streaming=streaming
        )

    # Disable automatic audio decoding to avoid torchcodec dependency
    # Audio will be decoded manually using librosa in AudioProcessor
    if not decode_audio:
        ds = ds.cast_column("audio", Audio(decode=False))

    return ds


def get_dataset_columns(dataset_key: str) -> dict:
    """
    Get audio and text column names for a dataset.

    Args:
        dataset_key: Key from HEBREW_DATASETS

    Returns:
        Dict with 'audio_col' and 'text_col' keys
    """
    if dataset_key in HEBREW_DATASETS:
        config = HEBREW_DATASETS[dataset_key]
        return {
            "audio_col": config["audio_col"],
            "text_col": config["text_col"],
        }
    # Default column names
    return {
        "audio_col": "audio",
        "text_col": "text",
    }
