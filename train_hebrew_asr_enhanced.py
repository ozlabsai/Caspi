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
from datasets import load_dataset, concatenate_datasets, Audio, Dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import evaluate
import wandb


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

    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = None

    # Training hyperparameters (Round 2: optimized for 2x A100)
    batch_size: int = 2  # Down from 8 for 2x A100
    gradient_accumulation_steps: int = 16  # Up from 4, effective batch = 64
    learning_rate: float = 5e-5  # Down from 2e-4, more conservative
    num_epochs: int = 5  # Up from 3 for gradual unfreezing
    warmup_steps: int = 500

    # Optimization
    fp16: bool = False
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Evaluation
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 50
    report_wer_and_cer: bool = True

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                "ivrit-ai/crowd-transcribe-v5",
                "ivrit-ai/crowd-recital-whisper-training"
            ]
        if self.lora_target_modules is None:
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]
        if self.duration_buckets is None:
            # Per training-suggestion.md Phase 1, section 3
            self.duration_buckets = {
                "short": (0.5, 5.0),
                "medium": (5.0, 15.0),
                "long": (15.0, 30.0),
            }


class HebrewTextNormalizer:
    """
    Normalize Hebrew text for ASR training.

    Implements Phase 0 from training-suggestion.md:
    - Remove niqqud (diacritics)
    - Normalize punctuation
    - Handle numbers
    - Consistent Hebrew/English mixing
    """

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Hebrew niqqud Unicode range
        self.niqqud_pattern = re.compile(r'[\u0591-\u05C7]')

        # Whisper-style timestamp tokens (from recital dataset)
        self.timestamp_pattern = re.compile(r'<\|[\d.]+\|>')

        # Excessive punctuation
        self.punct_pattern = re.compile(r'([.,!?;:])\1+')

    def normalize(self, text: str) -> str:
        """Apply all normalization rules."""

        # 1. Remove Whisper timestamp tokens (Phase 1, section 2)
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
        """Load, clean, and prepare datasets."""
        print("=" * 60)
        print("Loading Hebrew ASR Datasets")
        print("=" * 60)

        datasets = []
        for dataset_name in self.config.datasets:
            try:
                print(f"\nLoading {dataset_name}...")
                ds = load_dataset(dataset_name, split="train")
                print(f"  ✓ Loaded: {len(ds)} examples")

                # Standardize columns
                ds = self._standardize_dataset(ds, dataset_name)

                # Cast audio to target sampling rate
                ds = ds.cast_column("audio", Audio(sampling_rate=self.target_sampling_rate))

                datasets.append(ds)

            except Exception as e:
                print(f"  ✗ Error: {e}")
                continue

        if not datasets:
            raise ValueError("No datasets loaded! Check access permissions.")

        # Combine datasets
        combined = concatenate_datasets(datasets)
        print(f"\n✓ Combined: {len(combined)} examples")

        # Apply text normalization
        print("\nNormalizing transcripts...")
        combined = combined.map(
            self._normalize_text,
            desc="Text normalization"
        )

        # Apply audio chunking
        print("\nChunking long audio...")
        combined = self._chunk_long_audio(combined)
        print(f"✓ After chunking: {len(combined)} examples")

        # Filter by duration
        combined = combined.filter(
            lambda x: self.config.min_audio_length_seconds <=
                     len(x["audio"]["array"]) / self.target_sampling_rate <=
                     self.config.max_audio_length_seconds,
            desc="Duration filtering"
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
        """Apply audio chunking with overlap."""

        chunked_examples = []

        for example in dataset:
            audio_array = example["audio"]["array"]
            transcript = example["text"]

            # Chunk this example
            chunks = self.audio_chunker.chunk_audio(audio_array, transcript)

            # Create new examples for each chunk
            for chunk_audio, chunk_text in chunks:
                chunked_examples.append({
                    "audio": {
                        "array": chunk_audio,
                        "sampling_rate": self.target_sampling_rate
                    },
                    "text": chunk_text
                })

        # Convert back to Dataset
        from datasets import Dataset as HFDataset
        return HFDataset.from_list(chunked_examples)

    def preprocess_function(self, examples: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess batch for training."""

        # Extract audio
        audio_arrays = [audio["array"] for audio in examples["audio"]]

        # Process through feature extractor
        inputs = self.processor.feature_extractor(
            audio_arrays,
            sampling_rate=self.target_sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        # Tokenize transcripts
        labels = self.processor.tokenizer(
            examples["text"],
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
    """Custom data collator for speech-to-text."""

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Separate inputs and labels
        input_features = [{"input_features": f["input_features"]} for f in features]
        labels = [f["labels"] for f in features]

        # Pad inputs
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

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

    # Compute metrics
    wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
    cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

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


def setup_lora_model(model, config: TrainingConfig):
    """Configure LoRA for training."""
    print("\nSetting up LoRA...")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"✓ Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model


def main():
    """Main training orchestration."""

    config = TrainingConfig()

    print("=" * 60)
    print("Enhanced Hebrew ASR Fine-tuning")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Features:")
    print(f"  ✓ Text normalization (Hebrew-specific)")
    print(f"  ✓ Timestamp token removal")
    print(f"  ✓ Audio chunking with {config.chunk_overlap_seconds}s overlap")
    print(f"  ✓ Duration range: {config.min_audio_length_seconds}-{config.max_audio_length_seconds}s")
    print(f"  ✓ WER + CER evaluation")
    print("=" * 60)

    # Initialize Weights & Biases experiment tracking
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "round2-gradual-unfreezing")

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
            "effective_batch_size": config.batch_size * config.gradient_accumulation_steps,
            "learning_rate": config.learning_rate,
            "num_epochs": config.num_epochs,
            "warmup_steps": config.warmup_steps,
            "max_audio_length": config.max_audio_length_seconds,

            # LoRA config
            "lora_r": config.lora_r,
            "lora_alpha": config.lora_alpha,
            "lora_dropout": config.lora_dropout,

            # Optimization
            "bf16": config.bf16,
            "gradient_checkpointing": config.gradient_checkpointing,
            "group_by_length": True,
            "max_grad_norm": 1.0,
            "optimizer": "adamw_torch_fused",

            # Phase 0 audit results
            "phase0_low_quality_pct": phase0_results["low_quality_percentage"] if phase0_results else None,
            "phase0_decision": phase0_results["decision"] if phase0_results else None,
            "phase0_mean_coverage": phase0_results["coverage_distribution"]["mean"] if phase0_results else None,
        },
        tags=["round2", "gradual-unfreezing", "2xA100", "hebrew-asr"],
        notes=f"Round 2 training with gradual unfreezing (Strategy B→A). Effective batch: {config.batch_size * config.gradient_accumulation_steps}",
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
    processor = AutoProcessor.from_pretrained(config.model_name)

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map="auto",
        trust_remote_code=True,
    )

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # Round 2: Apply Strategy B freezing before LoRA
    model = setup_round2_freezing_strategy_b(model)

    model = setup_lora_model(model, config)

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

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor)

    # Metrics
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    # Training arguments (Round 2: with length bucketing and grad clipping)
    training_args = Seq2SeqTrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
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
        # Round 2 additions:
        group_by_length=True,  # Simple length-based bucketing
        max_grad_norm=1.0,     # Gradient clipping for stability
        optim="adamw_torch_fused",  # Faster fused optimizer
        weight_decay=0.01,     # Regularization
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

    # Final evaluation
    print("\nFinal evaluation...")
    metrics = trainer.evaluate()
    print(f"✓ Final WER: {metrics['eval_wer']:.2f}%")
    print(f"✓ Final CER: {metrics['eval_cer']:.2f}%")

    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Model saved to: {config.output_dir}")
    print("=" * 60)

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    main()
