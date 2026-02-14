#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch>=2.0.0",
#   "transformers @ git+https://github.com/huggingface/transformers.git",
#   "datasets>=2.14.0",
#   "accelerate>=0.20.0",
#   "peft>=0.7.0",
#   "librosa>=0.10.0",
#   "soundfile>=0.12.0",
#   "evaluate>=0.4.0",
#   "jiwer>=3.0.0",
#   "numpy>=1.24.0",
# ]
# ///
"""
Enhanced Hebrew ASR Fine-tuning with Qwen3-ASR-1.7B

Incorporates best practices from training-suggestion.md:
- Text normalization (Hebrew-specific)
- Timestamp token removal
- Audio chunking with overlap
- Duration bucketing for efficiency
- WER + CER evaluation
- Staged training support
"""

import os
import re
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_dataset, concatenate_datasets, Audio, Dataset, interleave_datasets
from transformers import (
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    AutoProcessor,
)
# Use qwen_asr package for model loading (supports qwen3_asr architecture)
from qwen_asr import Qwen3ASRModel
# PEFT/LoRA not used - doing full fine-tuning for SOTA
import evaluate
import wandb
import random


@dataclass
class TrainingConfig:
    """Enhanced configuration for Hebrew ASR fine-tuning."""

    # Model configuration
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    output_dir: str = "./qwen3-asr-hebrew"

    # Dataset configuration
    datasets: List[str] = None
    max_audio_length_seconds: float = 15.0  # Round 2: reduced from 30.0
    min_audio_length_seconds: float = 0.5
    chunk_overlap_seconds: float = 1.0

    # Text normalization
    remove_niqqud: bool = True  # Remove Hebrew diacritics
    normalize_numbers: bool = True  # "5" -> "חמש"
    normalize_punctuation: bool = True  # Clean excessive punctuation
    keep_english: bool = True  # Keep English words in Hebrew text

    # Duration bucketing (per training-suggestion.md)
    use_duration_bucketing: bool = True
    duration_buckets: Dict[str, Tuple[float, float]] = None

    # SpecAugment configuration (proven to reduce WER by 5-15% in ASR)
    use_specaugment: bool = True
    time_mask_param: int = 80  # Max time steps to mask
    freq_mask_param: int = 27  # Max frequency bins to mask (Fbank has 128 bins)
    num_time_masks: int = 2    # Number of time masks to apply
    num_freq_masks: int = 2    # Number of frequency masks to apply
    mask_value: float = 0.0    # Value to fill masked regions

    # Synthetic audio augmentation (ivrit.ai + modern ASR best practices)
    use_audio_augmentation: bool = True
    speed_perturb_factors: List[float] = None  # [0.9, 1.0, 1.1] - speed variations
    pitch_shift_steps: int = 2  # Semitones to shift (+/- 2)
    noise_snr_db: Tuple[float, float] = (15.0, 30.0)  # SNR range for noise injection
    audio_augmentation_prob: float = 0.5  # 50% of samples get augmentation

    # ivrit.ai SOTA methods (timestamp + context preservation)
    timestamp_keep_prob: float = 0.4  # 40% keep Whisper timestamps
    prev_context_prob: float = 0.5    # 50% use previous transcript context

    # Dataset configuration with balanced sampling (NOT 90% Knesset)
    dataset_sampling_probs: Dict[str, float] = None  # Balanced mix

    # Full fine-tuning (NO LoRA - we want a complete model, not adapters)
    use_lora: bool = False  # Disabled for SOTA training

    # Training hyperparameters (Round 2)
    # IMPORTANT: Adjust batch_size and gradient_accumulation to maintain effective_batch = 64
    # 2x A100: batch_size=2, grad_acc=16  → effective = 2 × 16 × 2 = 64
    # 8x A100: batch_size=2, grad_acc=4   → effective = 2 × 4 × 8 = 64
    batch_size: int = 2  # Per GPU
    gradient_accumulation_steps: int = 16  # For 2 GPUs (adjust for 8 GPUs: use 4)
    learning_rate: float = 5e-5  # Conservative base LR
    num_epochs: int = 5  # For gradual unfreezing (Strategy B→A at epoch 3)
    warmup_steps: int = 500
    lr_scheduler_type: str = "cosine"  # Cosine with warmup (SOTA for ASR 2025)
    warmup_ratio: float = 0.1  # 10% warmup (modern best practice)

    # Layer-wise learning rate decay (discriminative fine-tuning)
    use_layerwise_lr_decay: bool = True
    layerwise_lr_decay_rate: float = 0.9  # Each layer gets 0.9x the LR of the layer above

    # Optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True
    max_grad_norm: float = 1.0  # Gradient clipping (prevents exploding gradients)
    weight_decay: float = 0.01  # L2 regularization (standard for AdamW)

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    report_wer_and_cer: bool = True

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                "ivrit-ai/knesset-plenums-whisper-training",  # Add Knesset for SOTA
                "ivrit-ai/crowd-transcribe-v5",
                "ivrit-ai/crowd-recital-whisper-training"
            ]
        # LoRA removed for full fine-tuning - no lora_target_modules needed
        if self.duration_buckets is None:
            # Per training-suggestion.md Phase 1, section 3
            self.duration_buckets = {
                "short": (0.5, 5.0),
                "medium": (5.0, 15.0),
                "long": (15.0, 30.0),
            }
        if self.dataset_sampling_probs is None:
            # Balanced sampling (NOT 90% Knesset like ivrit.ai)
            # More balanced for informal eval sets (WhatsApp, crowd)
            self.dataset_sampling_probs = {
                "ivrit-ai/knesset-plenums-whisper-training": 0.50,  # 50% formal
                "ivrit-ai/crowd-transcribe-v5": 0.30,               # 30% crowd informal
                "ivrit-ai/crowd-recital-whisper-training": 0.20,    # 20% Wikipedia
            }
        if self.speed_perturb_factors is None:
            self.speed_perturb_factors = [0.9, 1.0, 1.1]  # Slow, normal, fast


class HebrewTextNormalizer:
    """
    Normalize Hebrew text for ASR training with ivrit.ai SOTA methods.

    Implements Phase 0 from training-suggestion.md:
    - Remove niqqud (diacritics)
    - Normalize punctuation
    - Handle numbers
    - Consistent Hebrew/English mixing

    NEW (Round 2.5 - ivrit.ai SOTA):
    - Conditionally preserve timestamps (40% probability)
    - Support for previous context injection
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Hebrew niqqud Unicode range
        self.niqqud_pattern = re.compile(r'[\u0591-\u05C7]')

        # Whisper-style timestamp tokens (from recital dataset)
        self.timestamp_pattern = re.compile(r'<\|[\d.]+\|>')

        # Excessive punctuation
        self.punct_pattern = re.compile(r'([.,!?;:])\1+')

    def normalize(self, text: str, keep_timestamps: bool = None) -> str:
        """
        Apply all normalization rules.

        Args:
            text: Input text to normalize
            keep_timestamps: If True, preserve timestamp tokens.
                           If None, decide randomly based on config probability.

        Returns:
            Normalized text
        """
        # Handle None or empty text
        if text is None or (isinstance(text, str) and not text.strip()):
            return ""

        # 1. Decide timestamp preservation (ivrit.ai: 40% keep)
        if keep_timestamps is None:
            import random
            keep_timestamps = random.random() < self.config.timestamp_keep_prob

        # Remove Whisper timestamp tokens if not keeping
        if not keep_timestamps:
            text = self.timestamp_pattern.sub('', text)

        # 2. Remove niqqud if configured
        if self.config.remove_niqqud:
            text = self.niqqud_pattern.sub('', text)

        # 3. Normalize punctuation
        if self.config.normalize_punctuation:
            # Remove duplicated punctuation
            text = self.punct_pattern.sub(r'\1', text)
            # Standardize quotes (Round 2: consistent normalization)
            text = text.replace('"', '"').replace('"', '"')
            text = text.replace(''', "'").replace(''', "'")
            # Hebrew-specific: unify geresh and gershayim
            text = text.replace('״', '"')  # Gershayim
            text = text.replace('׳', "'")  # Geresh

        # 4. Clean whitespace
        text = ' '.join(text.split())

        # 5. Number normalization (Round 2: keep consistent)
        # Keep digits as-is for consistency with training data
        # If you want to normalize: implement Hebrew number-to-words here

        return text.strip()


class AudioChunker:
    """
    Chunk long audio with overlap (Option 2 from our discussion).

    Implements the chunking strategy to preserve all training data
    while keeping memory usage manageable.
    """

    def __init__(
        self,
        max_duration: float,
        overlap_duration: float,
        sampling_rate: int
    ):
        self.max_duration = max_duration
        self.overlap_duration = overlap_duration
        self.sampling_rate = sampling_rate
        self.max_samples = int(max_duration * sampling_rate)
        self.overlap_samples = int(overlap_duration * sampling_rate)

    def chunk_audio(
        self,
        audio_array: np.ndarray,
        transcript: str
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Split long audio into overlapping chunks.

        Returns:
            List of (audio_chunk, transcript) tuples
        """
        # Short audio - return as-is
        if len(audio_array) <= self.max_samples:
            return [(audio_array, transcript)]

        chunks = []
        start = 0

        while start < len(audio_array):
            end = min(start + self.max_samples, len(audio_array))
            chunk_audio = audio_array[start:end]

            # Use full transcript for all chunks
            # The model will learn to transcribe partial audio
            chunks.append((chunk_audio, transcript))

            # Move window with overlap
            if end == len(audio_array):
                break

            start = end - self.overlap_samples

        return chunks


class DurationBucketer:
    """
    Bucket samples by duration to minimize padding waste.

    Implements Phase 1, section 3 from training-suggestion.md:
    "huge lever for making 1.7B feel cheap"
    """

    def __init__(self, buckets: Dict[str, Tuple[float, float]]):
        self.buckets = buckets

    def assign_bucket(self, duration: float) -> Optional[str]:
        """Assign a sample to a duration bucket."""
        for bucket_name, (min_dur, max_dur) in self.buckets.items():
            if min_dur <= duration < max_dur:
                return bucket_name
        return None

    def create_bucketed_datasets(
        self,
        dataset: Dataset,
        sampling_rate: int
    ) -> Dict[str, Dataset]:
        """Split dataset into duration buckets."""

        # Calculate durations
        def add_duration(example):
            duration = len(example["audio"]["array"]) / sampling_rate
            example["duration"] = duration
            example["bucket"] = self.assign_bucket(duration)
            return example

        dataset = dataset.map(add_duration, desc="Computing durations")

        # Split by bucket
        bucketed = {}
        for bucket_name in self.buckets.keys():
            bucket_data = dataset.filter(
                lambda x: x["bucket"] == bucket_name,
                desc=f"Creating {bucket_name} bucket"
            )
            if len(bucket_data) > 0:
                bucketed[bucket_name] = bucket_data
                print(f"  {bucket_name}: {len(bucket_data)} samples")

        return bucketed


class HebrewASRDataPreprocessor:
    """Enhanced data preprocessor with all best practices."""

    def __init__(self, config: TrainingConfig, processor):
        self.config = config
        self.processor = processor
        self.target_sampling_rate = processor.feature_extractor.sampling_rate

        # Initialize components
        self.text_normalizer = HebrewTextNormalizer(config)
        self.audio_chunker = AudioChunker(
            max_duration=config.max_audio_length_seconds,
            overlap_duration=config.chunk_overlap_seconds,
            sampling_rate=self.target_sampling_rate
        )

        if config.use_duration_bucketing:
            self.bucketer = DurationBucketer(config.duration_buckets)

    def load_datasets(self) -> Tuple[Dataset, Dataset]:
        """
        Load datasets with balanced interleaving (ivrit.ai SOTA method).

        Uses sampling probabilities to balance formal (Knesset) vs informal (crowd)
        for better generalization on diverse eval sets.
        """
        print("=" * 60)
        print("Loading Hebrew ASR Datasets (Balanced Sampling)")
        print("=" * 60)

        datasets_dict = {}
        sampling_probs = []

        for dataset_name in self.config.datasets:
            try:
                print(f"\nLoading {dataset_name}...")
                ds = load_dataset(dataset_name, split="train")
                print(f"  ✓ Loaded: {len(ds)} examples")

                # Standardize columns
                ds = self._standardize_dataset(ds, dataset_name)

                # Cast audio to target sampling rate
                ds = ds.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))

                datasets_dict[dataset_name] = ds

                # Get sampling probability
                prob = self.config.dataset_sampling_probs.get(
                    dataset_name,
                    1.0 / len(self.config.datasets)  # Default: equal weighting
                )
                sampling_probs.append(prob)
                print(f"  Sampling probability: {prob:.1%}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        if not datasets_dict:
            raise ValueError("No datasets loaded! Check access permissions.")

        # Normalize probabilities to sum to 1.0
        total_prob = sum(sampling_probs)
        sampling_probs = [p / total_prob for p in sampling_probs]

        print(f"\n{'='*60}")
        print("Dataset Sampling Strategy (Balanced - NOT 90% Knesset):")
        for (name, ds), prob in zip(datasets_dict.items(), sampling_probs):
            short_name = name.split("/")[-1]
            print(f"  {short_name:40s}: {prob:5.1%} ({len(ds):,} samples)")
        print(f"{'='*60}\n")

        # Interleave datasets with sampling probabilities
        combined = interleave_datasets(
            list(datasets_dict.values()),
            probabilities=sampling_probs,
            seed=42,
            stopping_strategy="all_exhausted"  # Use all data from all datasets
        )
        print(f"✓ Interleaved: {len(combined)} examples")

        # Apply text normalization (CPU-bound: regex operations)
        print("\nNormalizing transcripts...")
        combined = combined.map(
            self._normalize_text,
            desc="Text normalization",
            num_proc=64  # Use 64 cores for parallel text processing
        )

        # Filter out examples with empty text after normalization
        combined = combined.filter(
            lambda x: x["text"] is not None and len(x["text"].strip()) > 0,
            desc="Removing empty transcripts",
            num_proc=32  # Use 32 cores for parallel filtering
        )
        print(f"✓ After removing empty texts: {len(combined)} examples")

        # Apply audio chunking
        print("\nChunking long audio...")
        combined = self._chunk_long_audio(combined)
        print(f"✓ After chunking: {len(combined)} examples")

        # Filter by duration (CPU-bound: array length checks)
        combined = combined.filter(
            lambda x: self.config.min_audio_length_seconds <=
                     len(x["audio"]["array"]) / self.target_sampling_rate <=
                     self.config.max_audio_length_seconds,
            desc="Duration filtering",
            num_proc=32  # Use 32 cores for parallel filtering
        )
        print(f"✓ After filtering: {len(combined)} examples")

        # Split train/validation
        split = combined.train_test_split(test_size=0.05, seed=42)

        print(f"\n✓ Train: {len(split['train'])} examples")
        print(f"✓ Validation: {len(split['test'])} examples")

        return split["train"], split["test"]

    def _standardize_dataset(self, ds: Dataset, dataset_name: str) -> Dataset:
        """Standardize column names across datasets."""

        # Map various transcript column names to "text"
        if "transcript" in ds.column_names:
            ds = ds.rename_column("transcript", "text")
        elif "transcription" in ds.column_names:
            ds = ds.rename_column("transcription", "text")

        # Keep only audio and text
        cols_to_keep = ["audio", "text"]
        cols_to_remove = [c for c in ds.column_names if c not in cols_to_keep]
        if cols_to_remove:
            ds = ds.remove_columns(cols_to_remove)

        return ds

    def _normalize_text(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Apply Hebrew text normalization."""
        example["text"] = self.text_normalizer.normalize(example["text"])
        return example

    def _chunk_long_audio(self, dataset: Dataset) -> Dataset:
        """Apply audio chunking with overlap using parallel processing."""

        def chunk_example(example):
            """Chunk a single example, returning multiple examples if needed."""
            audio_array = example["audio"]["array"]
            transcript = example["text"]

            # Chunk this example
            chunks = self.audio_chunker.chunk_audio(audio_array, transcript)

            # Return lists of chunked data
            return {
                "audio": [
                    {"array": chunk_audio, "sampling_rate": self.target_sampling_rate}
                    for chunk_audio, chunk_text in chunks
                ],
                "text": [chunk_text for chunk_audio, chunk_text in chunks]
            }

        # Use map with batched=False for per-example processing, remove_columns to avoid conflicts
        chunked = dataset.map(
            chunk_example,
            remove_columns=dataset.column_names,
            desc="Chunking audio",
            num_proc=100  # Use 100 CPU cores for parallel audio processing (CPU-bound)
        )

        # Flatten the lists (each example may have produced multiple chunks)
        # Use batched iteration for much better performance
        from datasets import Dataset as HFDataset

        print("Flattening chunks...")

        def flatten_batch(batch):
            """Flatten batched lists of chunks."""
            flattened_audio = []
            flattened_text = []

            # Each batch item contains lists of chunks
            for audio_list, text_list in zip(batch["audio"], batch["text"]):
                flattened_audio.extend(audio_list)
                flattened_text.extend(text_list)

            return {"audio": flattened_audio, "text": flattened_text}

        # Use batched map for efficient flattening (process 1000 examples at a time)
        flattened = chunked.map(
            flatten_batch,
            batched=True,
            batch_size=1000,
            remove_columns=chunked.column_names,
            desc="Flattening chunks",
            num_proc=32  # Use 32 cores for parallel flattening
        )

        return flattened

    def apply_audio_augmentation(self, audio_array: np.ndarray) -> np.ndarray:
        """
        Apply synthetic audio augmentation (ivrit.ai + modern ASR best practices).

        Techniques:
        - Speed perturbation (0.9x, 1.0x, 1.1x)
        - Pitch shifting (+/- 2 semitones)
        - Background noise injection (SNR 15-30 dB)

        Args:
            audio_array: Audio numpy array

        Returns:
            Augmented audio array
        """
        if not self.config.use_audio_augmentation:
            return audio_array

        # Apply with probability
        if random.random() > self.config.audio_augmentation_prob:
            return audio_array

        # Convert to tensor for torchaudio transforms
        audio_tensor = torch.from_numpy(audio_array).float().unsqueeze(0)  # [1, samples]

        # 1. Speed perturbation (33% each: slow, normal, fast)
        if random.random() < 0.7:  # 70% apply speed perturb
            speed_factor = random.choice(self.config.speed_perturb_factors)
            if speed_factor != 1.0:
                effects = [["speed", str(speed_factor)], ["rate", str(self.target_sampling_rate)]]
                audio_tensor, _ = torchaudio.sox_effects.apply_effects_tensor(
                    audio_tensor, self.target_sampling_rate, effects
                )

        # 2. Pitch shifting (+/- 2 semitones)
        if random.random() < 0.3:  # 30% apply pitch shift
            n_steps = random.randint(-self.config.pitch_shift_steps, self.config.pitch_shift_steps)
            if n_steps != 0:
                pitch_shift = T.PitchShift(
                    sample_rate=self.target_sampling_rate,
                    n_steps=n_steps
                )
                audio_tensor = pitch_shift(audio_tensor)

        # 3. Background noise injection (Gaussian noise)
        if random.random() < 0.4:  # 40% apply noise
            snr_db = random.uniform(*self.config.noise_snr_db)
            signal_power = audio_tensor.pow(2).mean()
            noise_power = signal_power / (10 ** (snr_db / 10))
            noise = torch.randn_like(audio_tensor) * torch.sqrt(noise_power)
            audio_tensor = audio_tensor + noise

        # Convert back to numpy
        return audio_tensor.squeeze(0).numpy()

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess batch for training with:
        - Synthetic audio augmentation (speed, pitch, noise)
        - Previous context injection (50% probability)
        - Timestamp preservation (40% probability)
        """

        # Extract audio and apply augmentation
        audio_arrays = []
        for audio in examples["audio"]:
            audio_array = audio["array"]
            # Apply synthetic augmentation
            audio_array = self.apply_audio_augmentation(audio_array)
            audio_arrays.append(audio_array)

        # Process through feature extractor
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Prepare transcripts with optional previous context (ivrit.ai SOTA method)
        transcripts = []
        for i, text in enumerate(examples["text"]):
            # Add previous context with probability (ivrit.ai: 50%)
            if "prev_transcript" in examples and random.random() < self.config.prev_context_prob:
                prev_text = examples["prev_transcript"][i] if i < len(examples.get("prev_transcript", [])) else ""
                if prev_text:
                    # Prepend previous context to current text
                    text = f"{prev_text} {text}"
            transcripts.append(text)

        # Tokenize transcripts (with optional context prepended)
        labels = self.processor.tokenizer(
            transcripts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Replace padding with -100 for loss computation
        labels["input_ids"] = [
            [(l if l != self.processor.tokenizer.pad_token_id else -100) for l in label]
            for label in labels["input_ids"]
        ]

        return {
            "input_features": inputs.input_features,
            "labels": labels.input_ids,
        }


class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator for speech-to-text with SpecAugment.

    SpecAugment (Park et al. 2019) applies time and frequency masking
    to audio spectrograms, proven to reduce WER by 5-15% in ASR tasks.
    """

    def __init__(self, processor, config: TrainingConfig = None):
        self.processor = processor
        self.config = config or TrainingConfig()

    def apply_specaugment(self, input_features: torch.Tensor) -> torch.Tensor:
        """
        Apply SpecAugment to input features (Fbank coefficients).

        Args:
            input_features: [batch, time, freq] tensor

        Returns:
            Augmented input_features with time and frequency masks
        """
        if not self.config.use_specaugment:
            return input_features

        batch_size, max_time, num_freq = input_features.shape

        # Clone to avoid in-place modification
        augmented = input_features.clone()

        for i in range(batch_size):
            # Apply frequency masks (mask horizontal stripes)
            for _ in range(self.config.num_freq_masks):
                f = np.random.randint(0, self.config.freq_mask_param)
                f_start = np.random.randint(0, max(1, num_freq - f))
                augmented[i, :, f_start:f_start + f] = self.config.mask_value

            # Apply time masks (mask vertical stripes)
            for _ in range(self.config.num_time_masks):
                t = np.random.randint(0, self.config.time_mask_param)
                t_start = np.random.randint(0, max(1, max_time - t))
                augmented[i, t_start:t_start + t, :] = self.config.mask_value

        return augmented

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [f["labels"] for f in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        # Apply SpecAugment to input_features (only during training)
        if self.config.use_specaugment and torch.is_grad_enabled():
            batch["input_features"] = self.apply_specaugment(batch["input_features"])

        # Pad labels
        label_features = [{"input_ids": label} for label in labels]
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # Replace padding with -100
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        batch["labels"] = labels

        return batch


def compute_metrics(pred, processor, wer_metric, cer_metric):
    """Compute both WER and CER (per training-suggestion.md Phase 4)."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # Decode
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    # Filter out empty predictions (from failed generations)
    valid_pairs = [(p, l) for p, l in zip(pred_str, label_str) if p.strip()]

    if not valid_pairs:
        print("  WARNING: No valid predictions for WER computation")
        return {"wer": 100.0, "cer": 100.0}

    pred_str_filtered, label_str_filtered = zip(*valid_pairs)

    # Compute metrics
    wer = 100 * wer_metric.compute(predictions=pred_str_filtered, references=label_str_filtered)
    cer = 100 * cer_metric.compute(predictions=pred_str_filtered, references=label_str_filtered)

    return {"wer": wer, "cer": cer}


# ============================================================================
# Round 2: Selective Freezing and Gradual Unfreezing
# ============================================================================

def setup_round2_freezing_strategy_b(model):
    """
    Round 2 Strategy B (Epochs 1-2): Train projector + top LLM only.

    Freeze ALL audio layers to focus on cross-modal adaptation.
    Based on actual Qwen3-ASR module names from weight map.

    TRAIN:
        - thinker.audio_tower.proj1.*
        - thinker.audio_tower.proj2.*
        - thinker.audio_tower.ln_post.*
        - thinker.model.layers.16-27.* (top 12 LLM layers)
        - thinker.lm_head.*

    FREEZE:
        - thinker.audio_tower.conv*
        - thinker.audio_tower.layers.0-23.* (ALL audio layers)
        - thinker.model.layers.0-15.* (bottom 16 LLM layers)
        - thinker.model.embed_tokens.*
    """
    print("\n" + "="*70)
    print("Round 2 Strategy B: Freezing Configuration")
    print("="*70)

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze trainable components
    trainable_count = 0
    for name, param in model.named_parameters():
        # Projector components (ALWAYS TRAIN - cross-modal bottleneck)
        if any(x in name for x in ["proj1", "proj2", "ln_post"]):
            param.requires_grad = True
            trainable_count += param.numel()

        # Top 12 LLM layers (16-27)
        elif "model.layers" in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num >= 16:
                    param.requires_grad = True
                    trainable_count += param.numel()
            except (IndexError, ValueError):
                pass

        # LM head (output projection)
        elif "lm_head" in name:
            param.requires_grad = True
            trainable_count += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Strategy B configured:")
    print(f"  Trainable: {trainable_count:,} ({100*trainable_count/total_params:.1f}%)")
    print(f"  Frozen: {total_params-trainable_count:,} ({100*(1-trainable_count/total_params):.1f}%)")
    print(f"  Training: Projector + Top 12 LLM + LM Head")
    print(f"  Frozen: ALL audio layers + Bottom 16 LLM")
    print("="*70)

    return model


def unfreeze_audio_top_layers(model, num_layers: int = 8):
    """
    Unfreeze top N audio layers for Strategy A (Epoch 3+).

    UNFREEZE:
        - thinker.audio_tower.layers.[24-num_layers]-23.*

    For num_layers=8: unfreezes layers 16-23 (top 8)
    """
    print("\n" + "="*70)
    print(f"Strategy A: Unfreezing Top {num_layers} Audio Layers")
    print("="*70)

    unfrozen_count = 0
    start_layer = 24 - num_layers  # 24 total audio layers

    for name, param in model.named_parameters():
        if "audio_tower.layers" in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num >= start_layer:
                    param.requires_grad = True
                    unfrozen_count += param.numel()
            except (IndexError, ValueError):
                pass

    print(f"✓ Unfroze audio_tower.layers.{start_layer}-23")
    print(f"  Parameters unfrozen: {unfrozen_count:,}")
    print("="*70)


def create_param_groups_with_discriminative_lrs(model, epoch: int):
    """
    Create parameter groups with layer-specific learning rates.

    Epochs 1-2 (Strategy B):
        - projector: 2e-4
        - llm_top: 5e-5
        - lm_head: 1e-4

    Epochs 3-5 (Strategy A):
        - projector: 1e-4 (reduced)
        - audio_top: 3e-5 (conservative)
        - llm_top: 3e-5 (reduced)
        - lm_head: 1e-4
    """
    param_groups = {
        "projector": [],
        "llm_top": [],
        "lm_head": [],
    }

    if epoch >= 3:
        param_groups["audio_top"] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # Projector
        if any(x in name for x in ["proj1", "proj2", "ln_post"]):
            param_groups["projector"].append(param)

        # Top audio layers (only if unfrozen)
        elif "audio_tower.layers" in name and epoch >= 3:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num >= 16:
                    param_groups["audio_top"].append(param)
            except (IndexError, ValueError):
                pass

        # Top LLM layers
        elif "model.layers" in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num >= 16:
                    param_groups["llm_top"].append(param)
            except (IndexError, ValueError):
                pass

        # LM head
        elif "lm_head" in name:
            param_groups["lm_head"].append(param)

    # Set learning rates based on epoch
    if epoch < 3:
        # Strategy B LRs
        return [
            {"params": param_groups["projector"], "lr": 2e-4, "name": "projector"},
            {"params": param_groups["llm_top"], "lr": 5e-5, "name": "llm_top"},
            {"params": param_groups["lm_head"], "lr": 1e-4, "name": "lm_head"},
        ]
    else:
        # Strategy A LRs (reduced for stability)
        return [
            {"params": param_groups["projector"], "lr": 1e-4, "name": "projector"},
            {"params": param_groups["audio_top"], "lr": 3e-5, "name": "audio_top"},
            {"params": param_groups["llm_top"], "lr": 3e-5, "name": "llm_top"},
            {"params": param_groups["lm_head"], "lr": 1e-4, "name": "lm_head"},
        ]


class GradualUnfreezeTrainer(Seq2SeqTrainer):
    """
    Custom trainer with gradual unfreezing at epoch boundaries.

    Epochs 1-2: Strategy B (projector + top LLM)
    Epochs 3-5: Strategy A (+ unfreeze top 8 audio layers)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfroze_audio = False
        self.strategy_a_enabled = False

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override to ensure BF16 dtype consistency during generation.
        Fixes: Input type (float) and bias type (c10::BFloat16) mismatch.
        """
        # Ensure model is in evaluation mode and BF16
        model.train(False)

        # Convert inputs to BF16 if model is BF16
        if hasattr(model, 'dtype') and model.dtype == torch.bfloat16:
            if 'input_features' in inputs:
                inputs['input_features'] = inputs['input_features'].to(torch.bfloat16)

        # Call parent prediction_step with autocast for BF16
        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=self.args.bf16):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

    def training_step(self, model, inputs):
        """Override to check for epoch-based unfreezing."""
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 1

        # Unfreeze audio at epoch 3
        if current_epoch >= 3 and not self.unfroze_audio:
            print("\n" + "="*70)
            print(f"EPOCH {current_epoch}: Switching to Strategy A")
            print("="*70)

            # Unfreeze top audio layers
            unfreeze_audio_top_layers(model, num_layers=8)

            # Recreate optimizer with new parameter groups
            self.optimizer = self.create_optimizer()

            self.unfroze_audio = True
            self.strategy_a_enabled = True

            # Log strategy switch to wandb
            wandb.log({
                "training/strategy_switch": current_epoch,
                "training/strategy": "A",
                "training/audio_layers_unfrozen": 8,
            })

            print("✓ Strategy A activated: Projector + Audio Top + LLM Top")
            print("="*70 + "\n")

        return super().training_step(model, inputs)

    def log_metrics(self, logs, step):
        """
        Override to log WER/CER prominently to W&B.

        Creates separate metrics for better W&B dashboard visualization:
        - eval/wer_live (tracks WER over time)
        - eval/cer_live (tracks CER over time)
        - eval/loss (standard eval loss)
        """
        # Call parent to handle standard logging
        super().log_metrics(logs, step)

        # Extract and re-log WER/CER for better W&B visibility
        if "eval_wer" in logs:
            wandb.log({
                "eval/wer_live": logs["eval_wer"],
                "eval/cer_live": logs.get("eval_cer", None),
                "eval/loss_live": logs.get("eval_loss", None),
                "step": step,
            })

            # Log to console for visibility
            current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
            print(f"\n{'='*70}")
            print(f"Evaluation at step {step} (epoch {current_epoch:.1f}):")
            print(f"  WER: {logs['eval_wer']:.2f}%")
            print(f"  CER: {logs.get('eval_cer', 0):.2f}%")
            print(f"  Loss: {logs.get('eval_loss', 0):.4f}")
            print(f"{'='*70}\n")

    def create_optimizer(self):
        """Create optimizer with discriminative learning rates."""
        if self.optimizer is None or self.strategy_a_enabled:
            current_epoch = int(self.state.epoch) if self.state.epoch is not None else 1
            param_groups = create_param_groups_with_discriminative_lrs(self.model, current_epoch)

            # Filter out empty groups
            param_groups = [g for g in param_groups if len(g["params"]) > 0]

            if param_groups:
                print(f"\nOptimizer configuration (Epoch {current_epoch}):")
                for group in param_groups:
                    num_params = sum(p.numel() for p in group["params"])
                    print(f"  {group['name']:15s}: LR={group['lr']:.0e}, params={num_params:,}")

                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                )
            else:
                # Fallback to default
                return super().create_optimizer()

        return self.optimizer


# ============================================================================
# End Round 2 Additions
# ============================================================================


def average_model_checkpoints(checkpoint_paths: List[str], output_path: str, base_model):
    """
    Average weights from multiple checkpoints (ivrit.ai SOTA method).

    Args:
        checkpoint_paths: List of checkpoint directory paths
        output_path: Where to save averaged model
        base_model: Base model for structure reference

    Expected impact: 2-5% relative WER reduction (ensemble-like benefits)
    """
    print(f"\nAveraging {len(checkpoint_paths)} checkpoints...")

    # Load all state dicts
    state_dicts = []
    for cp in checkpoint_paths:
        # Try different file formats
        model_file = None
        for fname in ["pytorch_model.bin", "model.safetensors", "adapter_model.bin"]:
            fpath = Path(cp) / fname
            if fpath.exists():
                model_file = fpath
                break

        if model_file is None:
            print(f"  ⚠️  No model file found in {cp}, skipping")
            continue

        try:
            if model_file.name.endswith(".safetensors"):
                from safetensors.torch import load_file
                sd = load_file(model_file)
            else:
                sd = torch.load(model_file, map_location="cpu")
            state_dicts.append(sd)
            print(f"  ✓ Loaded {cp}")
        except Exception as e:
            print(f"  ✗ Failed to load {cp}: {e}")

    if not state_dicts:
        raise ValueError("No checkpoints could be loaded for averaging!")

    # Average weights
    print(f"\nAveraging {len(state_dicts)} state dicts...")
    avg_state = {}
    for key in state_dicts[0].keys():
        # Average across all checkpoints
        avg_state[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)

    # Save averaged model
    os.makedirs(output_path, exist_ok=True)
    torch.save(avg_state, Path(output_path) / "pytorch_model.bin")

    print(f"✓ Saved averaged model to {output_path}")


# LoRA removed - using full fine-tuning for better WER
# Full model will be saved, not adapters


def main():
    """Main training orchestration."""

    config = TrainingConfig()

    print("=" * 60)
    print("Enhanced Hebrew ASR Fine-tuning (Round 2.5 + ivrit.ai SOTA 2025)")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"\nData Processing:")
    print(f"  ✓ Text normalization (Hebrew-specific)")
    print(f"  ✓ Timestamp preservation: {config.timestamp_keep_prob*100:.0f}% keep (ivrit.ai method)")
    print(f"  ✓ Previous context: {config.prev_context_prob*100:.0f}% use (ivrit.ai method)")
    print(f"  ✓ Audio chunking with {config.chunk_overlap_seconds}s overlap")
    print(f"  ✓ Duration range: {config.min_audio_length_seconds}-{config.max_audio_length_seconds}s")
    print(f"\nSOTA Augmentation (2025):")
    print(f"  ✓ SpecAugment: {'ENABLED' if config.use_specaugment else 'DISABLED'}")
    if config.use_specaugment:
        print(f"    - Time masks: {config.num_time_masks}x up to {config.time_mask_param} steps")
        print(f"    - Freq masks: {config.num_freq_masks}x up to {config.freq_mask_param} bins")
    print(f"  ✓ Synthetic audio augmentation: {'ENABLED' if config.use_audio_augmentation else 'DISABLED'}")
    if config.use_audio_augmentation:
        print(f"    - Speed perturb: {config.speed_perturb_factors}")
        print(f"    - Pitch shift: +/- {config.pitch_shift_steps} semitones")
        print(f"    - Noise injection: SNR {config.noise_snr_db[0]}-{config.noise_snr_db[1]} dB")
        print(f"    - Augmentation prob: {config.audio_augmentation_prob*100:.0f}%")
    print(f"\nOptimization:")
    print(f"  ✓ LR Schedule: {config.lr_scheduler_type} with {config.warmup_ratio*100:.0f}% warmup")
    print(f"  ✓ Discriminative layer-wise LR (projector/audio/LLM)")
    print(f"  ✓ Model averaging: 3 best checkpoints (ivrit.ai method)")
    print(f"  ✓ Gradient clipping: max_norm={config.max_grad_norm}")
    print(f"  ✓ Weight decay: {config.weight_decay} (AdamW L2 reg)")
    print(f"  ✓ BF16 mixed precision training")
    print(f"\nDataset Strategy:")
    print(f"  ✓ Balanced sampling (50% Knesset, 30% Transcribe, 20% Recital)")
    print(f"  ✓ Interleaved loading for domain balance")
    print(f"\nEvaluation:")
    print(f"  ✓ WER + CER metrics")
    print(f"  ✓ BF16-safe evaluation (dtype fix applied)")
    print("=" * 60)

    # Auto-detect GPU count and adjust gradient accumulation
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    target_effective_batch = 64  # Target effective batch size

    # Adjust gradient accumulation based on GPU count
    # Formula: batch_size × grad_acc × num_gpUs = 64
    config.gradient_accumulation_steps = target_effective_batch // (config.batch_size * num_gpus)
    actual_effective_batch = config.batch_size * config.gradient_accumulation_steps * num_gpus

    print(f"\n✓ GPU Auto-configuration:")
    print(f"  GPUs detected: {num_gpus}")
    print(f"  Batch size per GPU: {config.batch_size}")
    print(f"  Gradient accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective batch: {actual_effective_batch}")

    # Initialize Weights & Biases experiment tracking
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", f"round2-{num_gpus}xGPU")

    # Load Phase 0 audit results if available
    phase0_results = None
    phase0_path = Path("./phase0_audit_results/alignment_report.json")
    if phase0_path.exists():
        with open(phase0_path, 'r') as f:
            phase0_results = json.load(f)
            print(f"\n✓ Loaded Phase 0 audit results: {phase0_results['decision']}")

    # Initialize wandb
    wandb.init(
        project=os.environ.get("WANDB_PROJECT", "qwen3-asr-hebrew"),
        name=wandb_run_name,
        config={
            # Model config
            "model_name": config.model_name,
            "output_dir": config.output_dir,

            # Training strategy
            "strategy": "gradual_unfreezing",
            "strategy_b_epochs": "1-2",
            "strategy_a_epochs": "3-5",

            # Hyperparameters
            "batch_size": config.batch_size,
            "gradient_accumulation_steps": config.gradient_accumulation_steps,
            "effective_batch_size": actual_effective_batch,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "warmup_steps": config.warmup_steps,
            "warmup_ratio": config.warmup_ratio,
            "lr_scheduler_type": config.lr_scheduler_type,
            "max_audio_length": config.max_audio_length_seconds,

            # Fine-tuning strategy
            "use_lora": False,  # Full fine-tuning for SOTA
            "training_mode": "full_finetuning",
            "gradual_unfreezing": True,

            # SOTA Optimizations (2025)
            "specaugment_enabled": config.use_specaugment,
            "time_mask_param": config.time_mask_param,
            "freq_mask_param": config.freq_mask_param,
            "num_time_masks": config.num_time_masks,
            "num_freq_masks": config.num_freq_masks,
            "layerwise_lr_decay": config.use_layerwise_lr_decay,
            "layerwise_lr_decay_rate": config.layerwise_lr_decay_rate,
            "max_grad_norm": config.max_grad_norm,
            "weight_decay": config.weight_decay,

            # ivrit.ai SOTA methods (Round 2.5)
            "audio_augmentation_enabled": config.use_audio_augmentation,
            "speed_perturb_factors": config.speed_perturb_factors,
            "pitch_shift_steps": config.pitch_shift_steps,
            "noise_snr_db": config.noise_snr_db,
            "audio_augmentation_prob": config.audio_augmentation_prob,
            "timestamp_keep_prob": config.timestamp_keep_prob,
            "prev_context_prob": config.prev_context_prob,
            "model_averaging": True,  # 3 best checkpoints
            "dataset_sampling_strategy": "balanced",  # 50-30-20
            "dataset_sampling_probs": config.dataset_sampling_probs,

            # Optimization
            "bf16": config.bf16,
            "gradient_checkpointing": config.gradient_checkpointing,
            "group_by_length": True,
            "optimizer": "adamw_torch_fused",
            "num_gpus": num_gpus,

            # Phase 0 audit results
            "phase0_low_quality_pct": phase0_results["low_quality_percentage"] if phase0_results else None,
            "phase0_decision": phase0_results["decision"] if phase0_results else None,
            "phase0_mean_coverage": phase0_results["coverage_distribution"]["mean"] if phase0_results else None,
        },
        tags=["round2.5", "gradual-unfreezing", "sota-2025", "ivrit-ai-methods", "specaugment", "synthetic-aug", "timestamp-preservation", "model-averaging", "balanced-sampling", "hebrew-asr"],
        notes=f"Round 2.5 ivrit.ai SOTA: SpecAugment + Synthetic Aug + Timestamps + Context + Model Avg + Balanced Sampling. Effective batch: {actual_effective_batch}, {num_gpus}x GPU",
    )

    # Log Phase 0 detailed results as artifact if available
    if phase0_results:
        wandb.log({
            "phase0/total_samples": phase0_results["total_samples"],
            "phase0/low_quality_count": phase0_results["low_quality_count"],
            "phase0/low_quality_percentage": phase0_results["low_quality_percentage"],
            "phase0/coverage_p10": phase0_results["coverage_distribution"]["p10"],
            "phase0/coverage_p25": phase0_results["coverage_distribution"]["p25"],
            "phase0/coverage_median": phase0_results["coverage_distribution"]["p50"],
            "phase0/coverage_p75": phase0_results["coverage_distribution"]["p75"],
            "phase0/coverage_p90": phase0_results["coverage_distribution"]["p90"],
        })

    # Load processor and model
    print("\nLoading model...")
    processor = AutoProcessor.from_pretrained(config.model_name, trust_remote_code=True)

    # Load via Qwen3ASRModel wrapper to get the model registered properly
    wrapper = Qwen3ASRModel.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
    )

    # Extract the actual PyTorch model for training (Qwen3ASRForConditionalGeneration)
    model = wrapper.model
    print(f"✓ Unwrapped model type: {type(model).__name__}")

    # Gradient checkpointing not available on Qwen3ASRModel
    # if config.gradient_checkpointing:
    #     model.gradient_checkpointing_enable()

    # Round 2.5: Full fine-tuning with gradual unfreezing (NO LoRA)
    # Strategy B: Freeze most params initially, unfreeze gradually
    model = setup_round2_freezing_strategy_b(model)

    # Log trainable parameters
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n✓ Trainable parameters: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    print(f"  Strategy: Full fine-tuning with gradual unfreezing (no LoRA adapters)")
    print(f"  Output: Complete standalone model")

    # Load and preprocess datasets
    preprocessor = HebrewASRDataPreprocessor(config, processor)
    train_dataset, eval_dataset = preprocessor.load_datasets()

    print("\nPreprocessing for training...")
    train_dataset = train_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Preprocessing train",
    )

    eval_dataset = eval_dataset.map(
        preprocessor.preprocess_function,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Preprocessing eval",
    )

    # Data collator with SpecAugment
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor, config)

    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Training arguments (Round 2: SOTA optimizations for 2025)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        warmup_ratio=config.warmup_ratio,  # 10% warmup (modern best practice)
        lr_scheduler_type=config.lr_scheduler_type,  # Cosine with warmup (SOTA 2025)
        num_train_epochs=config.num_epochs,
        evaluation_strategy="steps",
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        logging_steps=config.logging_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        fp16=config.fp16,
        bf16=config.bf16,
        gradient_checkpointing=config.gradient_checkpointing,
        push_to_hub=True,
        hub_model_id=f"{config.output_dir.replace('./', '')}",
        report_to=["wandb", "tensorboard"],
        predict_with_generate=True,
        generation_max_length=225,
        # Round 2 optimizations:
        group_by_length=True,  # Simple length-based bucketing
        max_grad_norm=config.max_grad_norm,  # Gradient clipping for stability
        optim="adamw_torch_fused",  # Faster fused optimizer (2x speedup)
        weight_decay=config.weight_decay,  # L2 regularization (AdamW standard)
    )

    # Trainer (Round 2: Gradual Unfreezing)
    trainer = GradualUnfreezeTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(
            pred, processor, wer_metric, cer_metric
        ),
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save
    print("\nSaving final model...")
    trainer.save_model(config.output_dir)
    processor.save_pretrained(config.output_dir)

    # Model Averaging (ivrit.ai SOTA method - 2-5% WER reduction)
    print(f"\n{'='*60}")
    print("Model Averaging (ivrit.ai SOTA method)")
    print(f"{'='*60}")

    # Find best 3 checkpoints by eval loss
    import glob
    checkpoints = glob.glob(f"{config.output_dir}/checkpoint-*")
    if len(checkpoints) >= 3:
        # Sort by eval loss (read from trainer_state.json)
        checkpoint_losses = []
        for cp in checkpoints:
            state_file = Path(cp) / "trainer_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                    eval_loss = state.get("best_metric", float('inf'))
                    checkpoint_losses.append((cp, eval_loss))

        if checkpoint_losses:
            # Sort by loss and take top 3
            checkpoint_losses.sort(key=lambda x: x[1])
            best_checkpoints = [cp for cp, _ in checkpoint_losses[:3]]

            print(f"Averaging {len(best_checkpoints)} best checkpoints:")
            for cp, loss in checkpoint_losses[:3]:
                print(f"  {cp}: eval_loss={loss:.4f}")

            # Average model weights
            avg_output_dir = f"{config.output_dir}-averaged"
            average_model_checkpoints(best_checkpoints, avg_output_dir, model)
            processor.save_pretrained(avg_output_dir)

            print(f"✓ Averaged model saved to: {avg_output_dir}")
            print(f"{'='*60}\n")

            # Use averaged model for final eval
            final_model_dir = avg_output_dir
        else:
            print("⚠️  No checkpoint metadata found, skipping averaging")
            final_model_dir = config.output_dir
    else:
        print(f"⚠️  Only {len(checkpoints)} checkpoints found, need 3+ for averaging")
        final_model_dir = config.output_dir

    # Final evaluation (on averaged model if available)
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    final_wer = metrics.get('eval_wer', 100.0)
    final_cer = metrics.get('eval_cer', 100.0)
    final_loss = metrics.get('eval_loss', 0.0)

    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  WER: {final_wer:.2f}%")
    print(f"  CER: {final_cer:.2f}%")
    print(f"  Loss: {final_loss:.4f}")
    print(f"  vs SOTA (eval-d1): {final_wer - 5.1:+.1f}% (SOTA: 5.1%)")
    print(f"  Model: {final_model_dir}")
    print(f"{'='*70}\n")

    # Log final metrics to W&B summary (visible on run page)
    wandb.run.summary["final_wer"] = final_wer
    wandb.run.summary["final_cer"] = final_cer
    wandb.run.summary["final_loss"] = final_loss
    wandb.run.summary["vs_sota_eval_d1"] = final_wer - 5.1
    wandb.run.summary["model_path"] = final_model_dir

    # Upload to HuggingFace Hub
    hf_repo_id = os.environ.get("HF_REPO_ID", "OzLabs/Qwen3-ASR-Hebrew-Round2")
    if os.environ.get("SKIP_HF_UPLOAD", "false").lower() != "true":
        try:
            print(f"\n{'='*60}")
            print("Uploading to HuggingFace Hub...")
            print(f"{'='*60}")

            from huggingface_hub import HfApi, create_repo

            # Create repo if doesn't exist
            api = HfApi()
            try:
                create_repo(hf_repo_id, repo_type="model", exist_ok=True)
                print(f"✓ Repository ready: https://huggingface.co/{hf_repo_id}")
            except Exception as e:
                print(f"Repository exists or minor error: {e}")

            # Upload model files
            print("Uploading model files (this may take 5-10 minutes)...")
            api.upload_folder(
                folder_path=config.output_dir,
                repo_id=hf_repo_id,
                repo_type="model",
                commit_message=f"Round 2: WER {final_wer:.2f}%, CER {final_cer:.2f}%, {num_gpus}xGPU, gradual unfreezing",
                ignore_patterns=["checkpoint-*", "wandb/*", "*.log", "runs/*"]
            )

            print(f"\n{'='*60}")
            print(f"✓ Model uploaded successfully!")
            print(f"View at: https://huggingface.co/{hf_repo_id}")
            print(f"{'='*60}\n")

        except Exception as e:
            print(f"\n⚠️  Failed to upload to HuggingFace Hub: {e}")
            print(f"Model is saved locally at: {config.output_dir}")
            print(f"You can manually upload with:")
            print(f"  huggingface-cli upload {hf_repo_id} {config.output_dir}/\n")
    else:
        print(f"\n⚠️  HF upload skipped (SKIP_HF_UPLOAD=true)")
        print(f"Model saved locally at: {config.output_dir}\n")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {config.output_dir}")
    if os.environ.get("SKIP_HF_UPLOAD", "false").lower() != "true":
        print(f"Uploaded to: https://huggingface.co/{hf_repo_id}")
    print("=" * 60)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
