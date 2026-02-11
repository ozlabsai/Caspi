#!/usr/bin/env python3
"""
Enhanced Hebrew ASR Fine-tuning with Qwen3-ASR-1.7B

Based on the official qwen3_asr_sft.py pattern from:
  https://github.com/QwenLM/Qwen3-ASR/blob/main/finetuning/qwen3_asr_sft.py

Round 2 enhancements:
- Hebrew text normalization (niqqud removal, punctuation cleanup)
- Gradual unfreezing (Strategy B → A)
- Discriminative learning rates per layer group
- Weights & Biases experiment tracking
- Duration filtering
"""

import os
import re
import json
import shutil
import time
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import random
import librosa
import numpy as np
import torch
from datasets import load_dataset, concatenate_datasets, Audio, Dataset
from qwen_asr import Qwen3ASRModel
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
import evaluate
import wandb

logger = logging.getLogger(__name__)


# ============================================================================
# Official Qwen3-ASR forward patch (from qwen3_asr_sft.py)
# ============================================================================

def patch_outer_forward(model):
    """Patch the outer model's forward to route through thinker."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError(
            "Cannot patch forward: model has no `.thinker.forward`. "
            "Your qwen3_asr model may be incompatible."
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class TrainingConfig:
    """Configuration for Hebrew ASR fine-tuning."""

    # Model
    model_name: str = "Qwen/Qwen3-ASR-1.7B"
    output_dir: str = "./qwen3-asr-hebrew-round2"

    # Dataset
    datasets: List[str] = None
    max_audio_length_seconds: float = 15.0
    min_audio_length_seconds: float = 0.5

    # Text normalization
    remove_niqqud: bool = True
    normalize_punctuation: bool = True

    # Training hyperparameters (optimized for 2x A100-40GB with DDP)
    batch_size: int = 4
    gradient_accumulation_steps: int = 8  # effective batch = 4 × 8 × 2 GPUs = 64
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.02

    # Optimization
    bf16: bool = True
    gradient_checkpointing: bool = True

    # Evaluation & saving
    eval_steps: int = 500
    save_steps: int = 500
    logging_steps: int = 25

    def __post_init__(self):
        if self.datasets is None:
            self.datasets = [
                "ivrit-ai/crowd-transcribe-v5",
                "ivrit-ai/crowd-recital-whisper-training"
            ]


# ============================================================================
# Hebrew Text Normalization
# ============================================================================

class HebrewTextNormalizer:
    """Normalize Hebrew text for ASR training."""

    def __init__(self):
        self.niqqud_pattern = re.compile(r'[\u0591-\u05C7]')
        self.timestamp_pattern = re.compile(r'<\|[\d.]+\|>')
        self.punct_pattern = re.compile(r'([.,!?;:])\1+')

    def normalize(self, text: str) -> str:
        if text is None:
            return ""
        # Remove Whisper timestamp tokens
        text = self.timestamp_pattern.sub('', text)
        # Remove niqqud
        text = self.niqqud_pattern.sub('', text)
        # Normalize punctuation
        text = self.punct_pattern.sub(r'\1', text)
        text = text.replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2018', "'").replace('\u2019', "'")
        text = text.replace('\u05f4', '"')   # Gershayim
        text = text.replace('\u05f3', "'")   # Geresh
        # Clean whitespace
        text = ' '.join(text.split())
        return text.strip()


# ============================================================================
# Data loading from HF datasets → JSONL with WAV files
# ============================================================================

def load_and_prepare_data(config: TrainingConfig, normalizer: HebrewTextNormalizer, data_dir: str):
    """
    Load HF datasets, normalize text, save audio as WAV files,
    and create JSONL files for training in the official Qwen3-ASR format.
    """
    print("=" * 60)
    print("Loading Hebrew ASR Datasets")
    print("=" * 60)

    all_datasets = []
    for dataset_name in config.datasets:
        try:
            print(f"\nLoading {dataset_name}...")
            ds = load_dataset(dataset_name, split="train")
            print(f"  ✓ Loaded: {len(ds)} examples")

            # Standardize column names to "text"
            if "text" not in ds.column_names:
                for alt in ["sentence", "transcript", "transcription"]:
                    if alt in ds.column_names:
                        ds = ds.rename_column(alt, "text")
                        break

            # Keep only audio and text
            cols_to_remove = [c for c in ds.column_names if c not in ["audio", "text"]]
            if cols_to_remove:
                ds = ds.remove_columns(cols_to_remove)

            # Cast audio to 16kHz
            ds = ds.cast_column("audio", Audio(sampling_rate=16000))
            all_datasets.append(ds)

        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    if not all_datasets:
        raise ValueError("No datasets loaded!")

    combined = concatenate_datasets(all_datasets)
    print(f"\n✓ Combined: {len(combined)} examples")

    # Normalize text
    print("\nNormalizing transcripts...")
    def normalize_text(example):
        example["text"] = normalizer.normalize(example["text"])
        return example
    combined = combined.map(normalize_text, desc="Text normalization")

    # Filter empty text
    combined = combined.filter(lambda x: len(x["text"].strip()) > 0, desc="Removing empty")
    print(f"✓ After text cleanup: {len(combined)} examples")

    # Split train/eval
    split = combined.train_test_split(test_size=0.05, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]
    print(f"✓ Train: {len(train_ds)}, Eval: {len(eval_ds)}")

    # Save to WAV + JSONL
    os.makedirs(data_dir, exist_ok=True)
    wav_dir = os.path.join(data_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    train_jsonl = os.path.join(data_dir, "train.jsonl")
    eval_jsonl = os.path.join(data_dir, "eval.jsonl")

    for split_name, ds, jsonl_path in [("train", train_ds, train_jsonl), ("eval", eval_ds, eval_jsonl)]:
        if os.path.exists(jsonl_path):
            # Count existing lines
            with open(jsonl_path) as f:
                existing = sum(1 for _ in f)
            if existing == len(ds):
                print(f"✓ {split_name} JSONL already exists ({existing} examples), skipping export")
                continue
        print(f"\nExporting {split_name} to WAV + JSONL ({len(ds)} examples)...")
        import soundfile as sf
        written = 0
        skipped = 0
        with open(jsonl_path, 'w') as f:
            for i, example in enumerate(ds):
                try:
                    audio = example["audio"]
                    array = audio["array"]
                    sr = audio["sampling_rate"]
                    duration = len(array) / sr

                    # Duration filter
                    if duration < config.min_audio_length_seconds or duration > config.max_audio_length_seconds:
                        skipped += 1
                        continue

                    wav_path = os.path.join(wav_dir, f"{split_name}_{i:07d}.wav")
                    sf.write(wav_path, array, sr)

                    # Qwen3-ASR format: "language Hebrew<asr_text>TRANSCRIPT"
                    text = f"language Hebrew<asr_text>{example['text']}"
                    f.write(json.dumps({"audio": wav_path, "text": text}) + '\n')
                    written += 1

                    if (i + 1) % 10000 == 0:
                        print(f"  {i+1}/{len(ds)} exported ({skipped} skipped for duration)")
                except Exception as e:
                    skipped += 1
                    continue

        print(f"  ✓ {split_name}: {written} examples written, {skipped} skipped")

    return train_jsonl, eval_jsonl


# ============================================================================
# Official Qwen3-ASR data collator (adapted from qwen3_asr_sft.py)
# ============================================================================

def build_prefix_messages(prompt, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    def _preprocess(ex):
        prompt = ex.get("prompt", "")
        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio": ex["audio"],
            "target": ex["text"],
            "prefix_text": prefix_text,
        }
    return _preprocess


# ============================================================================
# Data Augmentation: SpecAugment + Speed Perturbation
# ============================================================================

def apply_spec_augment(
    input_features: torch.Tensor,
    feature_attention_mask: Optional[torch.Tensor] = None,
    freq_mask_param: int = 27,
    time_mask_param: int = 100,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    Apply SpecAugment to mel-spectrogram features.
    Input shape: (batch, n_mels=128, time_frames)
    """
    features = input_features.clone()
    batch_size, freq_size, time_size = features.shape

    for b in range(batch_size):
        # Get actual length for this sample (avoid masking padding)
        if feature_attention_mask is not None:
            actual_len = int(feature_attention_mask[b].sum().item())
        else:
            actual_len = time_size

        # Frequency masks
        for _ in range(num_freq_masks):
            f = random.randint(0, min(freq_mask_param, freq_size - 1))
            f0 = random.randint(0, freq_size - f)
            features[b, f0:f0 + f, :] = 0

        # Time masks
        for _ in range(num_time_masks):
            t = random.randint(0, min(time_mask_param, max(1, actual_len - 1)))
            t0 = random.randint(0, max(1, actual_len - t))
            features[b, :, t0:t0 + t] = 0

    return features


def apply_speed_perturbation(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Randomly change audio speed by 0.9x, 1.0x, or 1.1x."""
    speed = random.choice([0.9, 1.0, 1.0, 1.1])  # 50% unchanged, 25% slow, 25% fast
    if speed == 1.0:
        return audio
    return librosa.effects.time_stretch(audio, rate=speed)


@dataclass
class DataCollatorForQwen3ASR:
    """Data collator with SpecAugment + speed perturbation."""
    processor: Any
    sampling_rate: int = 16000
    augment: bool = True
    spec_augment: bool = True
    speed_perturb: bool = True

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        # Load audio with optional speed perturbation
        audios = []
        for p in audio_paths:
            audio, _ = librosa.load(p, sr=self.sampling_rate, mono=True)
            if self.augment and self.speed_perturb:
                audio = apply_speed_perturbation(audio, self.sampling_rate)
            audios.append(audio)

        full_inputs = self.processor(
            text=full_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios,
            return_tensors="pt", padding=True, truncation=False,
        )

        # Apply SpecAugment to input features
        if self.augment and self.spec_augment and "input_features" in full_inputs:
            full_inputs["input_features"] = apply_spec_augment(
                full_inputs["input_features"],
                feature_attention_mask=full_inputs.get("feature_attention_mask"),
            )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs


# ============================================================================
# Checkpoint callback (from official script)
# ============================================================================

def copy_required_hf_files(src_dir: str, dst_dir: str):
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json", "generation_config.json", "preprocessor_config.json",
        "processor_config.json", "tokenizer_config.json", "tokenizer.json",
        "special_tokens_map.json", "chat_template.json", "merges.txt", "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class CheckpointInferableCallback(TrainerCallback):
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return control
        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.isdir(ckpt_dir):
            copy_required_hf_files(self.base_model_path, ckpt_dir)
        return control


# ============================================================================
# Comprehensive Logging Callback
# ============================================================================

class ComprehensiveLoggingCallback(TrainerCallback):
    """Logs detailed training metrics to console and wandb."""

    def __init__(self):
        self.train_start_time = None
        self.step_start_time = None
        self.last_log_time = None
        self.current_strategy = "B"
        self.epoch_start_time = None
        self.last_epoch = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if args.process_index != 0:
            return
        self.train_start_time = time.time()
        self.last_log_time = self.train_start_time

        total_params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("\n" + "=" * 70)
        print("TRAINING STARTED")
        print("=" * 70)
        print(f"  Total parameters:     {total_params:>14,}")
        print(f"  Trainable parameters: {trainable:>14,} ({100*trainable/total_params:.1f}%)")
        print(f"  Total steps:          {state.max_steps:>14,}")
        print(f"  Steps per epoch:      {state.max_steps // args.num_train_epochs:>14,}")
        print(f"  Logging every:        {args.logging_steps:>14} steps")
        print(f"  Eval every:           {args.eval_steps or 'N/A':>14} steps")
        print(f"  Save every:           {args.save_steps:>14} steps")
        print(f"  Strategy:             {'B (projector + LLM top)':>14}")

        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}:                {torch.cuda.get_device_name(i)} ({mem:.0f}GB)")
        print("=" * 70 + "\n")

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if args.process_index != 0 or logs is None:
            return

        now = time.time()
        elapsed = now - self.train_start_time if self.train_start_time else 0
        step = state.global_step

        # GPU memory stats
        gpu_metrics = {}
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            peak = torch.cuda.max_memory_allocated(i) / 1e9
            gpu_metrics[f"gpu/{i}/mem_allocated_gb"] = round(alloc, 2)
            gpu_metrics[f"gpu/{i}/mem_reserved_gb"] = round(reserved, 2)
            gpu_metrics[f"gpu/{i}/mem_peak_gb"] = round(peak, 2)

        # Throughput
        if self.last_log_time and step > 0:
            time_since_last = now - self.last_log_time
            steps_since_last = args.logging_steps
            samples_per_sec = (steps_since_last * args.per_device_train_batch_size *
                               args.gradient_accumulation_steps *
                               max(1, torch.cuda.device_count())) / time_since_last
            gpu_metrics["throughput/samples_per_sec"] = round(samples_per_sec, 1)
            gpu_metrics["throughput/steps_per_sec"] = round(steps_since_last / time_since_last, 3)
            gpu_metrics["throughput/sec_per_step"] = round(time_since_last / steps_since_last, 2)

        # Per-group learning rates from optimizer
        optimizer = kwargs.get("optimizer")
        if optimizer is None and hasattr(state, "_trainer"):
            optimizer = getattr(state._trainer, "optimizer", None)
        if optimizer:
            for i, group in enumerate(optimizer.param_groups):
                name = group.get("name", f"group_{i}")
                gpu_metrics[f"lr/{name}"] = group["lr"]

        # Strategy tracking
        gpu_metrics["training/strategy"] = 1 if self.current_strategy == "A" else 0
        gpu_metrics["training/strategy_name"] = self.current_strategy

        # ETA
        if step > 0 and state.max_steps > 0:
            frac = step / state.max_steps
            total_est = elapsed / frac
            remaining = total_est - elapsed
            gpu_metrics["training/eta_hours"] = round(remaining / 3600, 2)
            gpu_metrics["training/elapsed_hours"] = round(elapsed / 3600, 2)
            gpu_metrics["training/progress_pct"] = round(frac * 100, 2)

        # Epoch tracking
        epoch = logs.get("epoch", 0)
        current_epoch_int = int(epoch) + 1
        if current_epoch_int != self.last_epoch:
            self.epoch_start_time = now
            self.last_epoch = current_epoch_int

        # Estimated per-token loss (avg ~20 target tokens per example)
        loss = logs.get("loss", None)
        if loss is not None:
            gpu_metrics["loss/per_token_est"] = round(loss / 20.0, 4)

        # Log to wandb
        wandb.log(gpu_metrics, step=step)

        # Console summary
        loss_str = f"loss={loss:.2f}" if loss else "loss=N/A"
        per_token = f"(~{loss/20:.2f}/tok)" if loss else ""
        grad = logs.get("grad_norm", None)
        grad_str = f"grad={grad:.1f}" if grad else ""
        lr = logs.get("learning_rate", None)
        lr_str = f"lr={lr:.2e}" if lr else ""
        epoch_str = f"ep={epoch:.3f}" if epoch else ""
        eta_str = ""
        if "training/eta_hours" in gpu_metrics:
            eta_h = gpu_metrics["training/eta_hours"]
            eta_str = f"ETA={eta_h:.1f}h"

        local_gpu = int(os.environ.get("LOCAL_RANK", 0))
        mem_alloc = torch.cuda.memory_allocated(local_gpu) / 1e9
        mem_peak = torch.cuda.max_memory_allocated(local_gpu) / 1e9
        mem_str = f"GPU{local_gpu}=[{mem_alloc:.1f}/{mem_peak:.1f}GB]"

        throughput_str = ""
        if "throughput/sec_per_step" in gpu_metrics:
            throughput_str = f"{gpu_metrics['throughput/sec_per_step']:.2f}s/step"

        parts = [f"[Step {step}/{state.max_steps}]",
                 loss_str, per_token, grad_str, lr_str,
                 epoch_str, f"strat={self.current_strategy}",
                 throughput_str, mem_str, eta_str]
        print("  ".join(p for p in parts if p))

        self.last_log_time = now

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if args.process_index != 0 or not metrics:
            return
        print("\n" + "-" * 70)
        print(f"EVAL @ Step {state.global_step}:")
        for k, v in sorted(metrics.items()):
            if isinstance(v, float):
                print(f"  {k}: {v:.4f}")
            else:
                print(f"  {k}: {v}")
        print("-" * 70 + "\n")

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        epoch = int(state.epoch)
        elapsed = time.time() - (self.epoch_start_time or self.train_start_time)
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch} COMPLETED in {elapsed/60:.1f} min")
        if epoch == 2:
            print(">>> Next epoch will switch to Strategy A (unfreezing audio layers)")
        print(f"{'='*70}\n")

    def on_train_end(self, args, state, control, **kwargs):
        if args.process_index != 0:
            return
        total = time.time() - self.train_start_time if self.train_start_time else 0
        print("\n" + "=" * 70)
        print(f"TRAINING COMPLETE")
        print(f"  Total time: {total/3600:.2f} hours")
        print(f"  Total steps: {state.global_step}")
        print(f"  Final loss: {state.log_history[-1].get('loss', 'N/A') if state.log_history else 'N/A'}")

        for i in range(torch.cuda.device_count()):
            peak = torch.cuda.max_memory_allocated(i) / 1e9
            print(f"  GPU {i} peak memory: {peak:.2f} GB")
        print("=" * 70 + "\n")


# ============================================================================
# WER/CER Evaluation Callback
# ============================================================================

class WERCERCallback(TrainerCallback):
    """Compute WER/CER on a subset of eval data at each evaluation step."""

    def __init__(self, processor, eval_jsonl_path: str, wer_metric, cer_metric,
                 num_samples: int = 100):
        self.processor = processor
        self.wer_metric = wer_metric
        self.cer_metric = cer_metric
        self.normalizer = HebrewTextNormalizer()

        # Load eval samples
        self.eval_samples = []
        with open(eval_jsonl_path) as f:
            for i, line in enumerate(f):
                if i >= num_samples:
                    break
                data = json.loads(line)
                # Extract clean text (remove "language Hebrew<asr_text>" prefix)
                text = data["text"]
                if "<asr_text>" in text:
                    text = text.split("<asr_text>", 1)[1]
                self.eval_samples.append({
                    "audio_path": data["audio"],
                    "reference": text,
                })

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if args.process_index != 0 or not self.eval_samples:
            return

        print(f"\n  Computing WER/CER on {len(self.eval_samples)} samples...")

        # Get unwrapped model
        unwrapped = model.module if hasattr(model, "module") else model
        unwrapped.eval()
        device = next(unwrapped.parameters()).device

        predictions = []
        references = []

        # Build prompt template once
        dummy_msgs = build_prefix_messages("", None)
        prompt_text = self.processor.apply_chat_template(
            [dummy_msgs], add_generation_prompt=True, tokenize=False
        )[0] + "language Hebrew<asr_text>"

        batch_size = 4
        for i in range(0, len(self.eval_samples), batch_size):
            batch = self.eval_samples[i:i + batch_size]
            try:
                audios = [
                    librosa.load(s["audio_path"], sr=16000, mono=True)[0]
                    for s in batch
                ]
                texts = [prompt_text] * len(audios)

                inputs = self.processor(
                    text=texts, audio=audios,
                    return_tensors="pt", padding=True, truncation=False,
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    generated = unwrapped.generate(
                        **inputs, max_new_tokens=256,
                    )

                # Decode only new tokens
                prompt_len = inputs["input_ids"].shape[1]
                new_tokens = generated[:, prompt_len:]
                decoded = self.processor.batch_decode(
                    new_tokens, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                for j, s in enumerate(batch):
                    pred = self.normalizer.normalize(decoded[j])
                    ref = self.normalizer.normalize(s["reference"])
                    if ref:  # skip empty references
                        predictions.append(pred)
                        references.append(ref)

            except Exception as e:
                print(f"  WER eval batch error: {e}")
                continue

        unwrapped.train()

        if not predictions:
            print("  No valid predictions for WER computation")
            return

        wer = self.wer_metric.compute(predictions=predictions, references=references)
        cer = self.cer_metric.compute(predictions=predictions, references=references)

        wandb.log({
            "eval/wer": wer,
            "eval/cer": cer,
            "eval/wer_pct": wer * 100,
            "eval/cer_pct": cer * 100,
            "eval/num_samples": len(predictions),
        }, step=state.global_step)

        print(f"  WER: {wer*100:.2f}%  CER: {cer*100:.2f}%  ({len(predictions)} samples)")


# ============================================================================
# Round 2: Selective Freezing and Gradual Unfreezing
# ============================================================================

def setup_strategy_b(model):
    """
    Strategy B (Epochs 1-2): Train projector + top 12 LLM layers + LM head.
    Freeze ALL audio layers + bottom 16 LLM layers.
    """
    print("\n" + "="*70)
    print("Round 2 Strategy B: Freezing Configuration")
    print("="*70)

    for param in model.parameters():
        param.requires_grad = False

    trainable_count = 0
    for name, param in model.named_parameters():
        if any(x in name for x in ["proj1", "proj2", "ln_post"]):
            param.requires_grad = True
            trainable_count += param.numel()
        elif "model.layers" in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num >= 16:
                    param.requires_grad = True
                    trainable_count += param.numel()
            except (IndexError, ValueError):
                pass
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
    """Unfreeze top N audio layers for Strategy A (Epoch 3+)."""
    print("\n" + "="*70)
    print(f"Strategy A: Unfreezing Top {num_layers} Audio Layers")
    print("="*70)

    unfrozen_count = 0
    start_layer = 24 - num_layers

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


def create_param_groups(model, epoch: int):
    """Create parameter groups with discriminative learning rates."""
    groups = {"projector": [], "llm_top": [], "lm_head": []}
    if epoch >= 3:
        groups["audio_top"] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(x in name for x in ["proj1", "proj2", "ln_post"]):
            groups["projector"].append(param)
        elif "audio_tower.layers" in name and epoch >= 3:
            groups["audio_top"].append(param)
        elif "model.layers" in name:
            try:
                layer_num = int(name.split("layers.")[1].split(".")[0])
                if layer_num >= 16:
                    groups["llm_top"].append(param)
            except (IndexError, ValueError):
                pass
        elif "lm_head" in name:
            groups["lm_head"].append(param)

    if epoch < 3:
        return [
            {"params": groups["projector"], "lr": 2e-4, "name": "projector"},
            {"params": groups["llm_top"], "lr": 5e-5, "name": "llm_top"},
            {"params": groups["lm_head"], "lr": 1e-4, "name": "lm_head"},
        ]
    else:
        return [
            {"params": groups["projector"], "lr": 1e-4, "name": "projector"},
            {"params": groups["audio_top"], "lr": 3e-5, "name": "audio_top"},
            {"params": groups["llm_top"], "lr": 3e-5, "name": "llm_top"},
            {"params": groups["lm_head"], "lr": 1e-4, "name": "lm_head"},
        ]


# ============================================================================
# Custom Trainer with gradual unfreezing + float casting
# ============================================================================

class GradualUnfreezeTrainer(Trainer):
    """
    Combines:
    - CastFloatInputsTrainer (from official script): casts float tensors to model dtype
    - Gradual unfreezing: Strategy B (epochs 1-2) → Strategy A (epochs 3-5)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unfroze_audio = False
        self.strategy_a_enabled = False

    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

    def training_step(self, model, inputs, num_items_in_batch=None):
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 1

        if current_epoch >= 3 and not self.unfroze_audio:
            print(f"\n{'='*70}")
            print(f"EPOCH {current_epoch}: Switching to Strategy A")
            print("="*70)

            unfreeze_audio_top_layers(model, num_layers=8)
            self.optimizer = self.create_optimizer()
            self.unfroze_audio = True
            self.strategy_a_enabled = True

            # Notify logging callback about strategy switch
            for cb in self.callback_handler.callbacks:
                if isinstance(cb, ComprehensiveLoggingCallback):
                    cb.current_strategy = "A"

            if int(os.environ.get("LOCAL_RANK", 0)) == 0:
                wandb.log({
                    "training/strategy_switch": current_epoch,
                    "training/strategy": "A",
                    "training/audio_layers_unfrozen": 8,
                })

                total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
                total_params = sum(p.numel() for p in model.parameters())
                print(f"  New trainable: {total_trainable:,} ({100*total_trainable/total_params:.1f}%)")

            print("✓ Strategy A activated")
            print("="*70 + "\n")

        return super().training_step(model, inputs, num_items_in_batch)

    def create_optimizer(self):
        if self.optimizer is None or self.strategy_a_enabled:
            current_epoch = int(self.state.epoch) if self.state.epoch is not None else 1
            param_groups = create_param_groups(self.model, current_epoch)
            param_groups = [g for g in param_groups if len(g["params"]) > 0]

            if param_groups:
                print(f"\nOptimizer (Epoch {current_epoch}):")
                for group in param_groups:
                    n = sum(p.numel() for p in group["params"])
                    print(f"  {group['name']:15s}: LR={group['lr']:.0e}, params={n:,}")

                self.optimizer = torch.optim.AdamW(
                    param_groups,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                )
            else:
                return super().create_optimizer()

        return self.optimizer


# ============================================================================
# Main
# ============================================================================

def main():
    config = TrainingConfig()

    # DDP rank awareness
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_main = local_rank == 0

    # Configure logging
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        level=logging.INFO if is_main else logging.WARNING,
        force=True,
    )

    if is_main:
        print("=" * 60)
        print("Round 2: Hebrew ASR Fine-tuning (Official Qwen3-ASR pattern)")
        print("=" * 60)
        print(f"Model: {config.model_name}")
        print(f"Output: {config.output_dir}")
        print(f"Batch: {config.batch_size} x {config.gradient_accumulation_steps} x 2 GPUs = "
              f"{config.batch_size * config.gradient_accumulation_steps * 2} effective")
        print(f"Duration: {config.min_audio_length_seconds}-{config.max_audio_length_seconds}s")
        print(f"Epochs: {config.num_epochs} (Strategy B→A at epoch 3)")
        print("=" * 60)

    # Initialize wandb (rank 0 only)
    wandb_run_name = os.environ.get("WANDB_RUN_NAME", "round2-gradual-unfreezing")
    if is_main:
        wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "OzLabs"),
            project=os.environ.get("WANDB_PROJECT", "qwen3-asr-hebrew"),
            name=wandb_run_name,
            config={
                "model_name": config.model_name,
                "strategy": "gradual_unfreezing_B_to_A",
                "batch_size": config.batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "effective_batch_size": config.batch_size * config.gradient_accumulation_steps * 2,
                "learning_rate": config.learning_rate,
                "num_epochs": config.num_epochs,
                "max_audio_length": config.max_audio_length_seconds,
                "bf16": config.bf16,
                "ddp": True,
                "num_gpus": 2,
                "spec_augment": True,
                "speed_perturbation": True,
                "wer_eval_samples": 100,
            },
            tags=["round2", "gradual-unfreezing", "2xA100", "hebrew-asr", "ddp", "augmented"],
        )
    # rank != 0 will use report_to=[] in TrainingArguments

    # Load model (official qwen-asr way)
    if is_main:
        print("\nLoading model...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        config.model_name,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,  # Required for DDP / multi-GPU
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    # Patch forward method (official pattern)
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # Resolve model path for checkpoint callback
    from huggingface_hub import snapshot_download
    try:
        model_local_path = snapshot_download(config.model_name, local_files_only=True)
    except Exception:
        model_local_path = config.model_name

    # Apply Strategy B freezing
    model = setup_strategy_b(model)

    # Pre-load metrics
    if is_main:
        print("\nLoading evaluation metrics...")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    if is_main:
        print("✓ Metrics loaded")

    # Prepare data (HF datasets → WAV + JSONL)
    normalizer = HebrewTextNormalizer()
    data_dir = "./qwen3_asr_round2_data"
    train_jsonl = os.path.join(data_dir, "train.jsonl")
    eval_jsonl = os.path.join(data_dir, "eval.jsonl")

    if os.path.exists(train_jsonl) and os.path.exists(eval_jsonl):
        if is_main:
            print(f"\n✓ Training data already exists, skipping export")
    else:
        if is_main:
            train_jsonl, eval_jsonl = load_and_prepare_data(config, normalizer, data_dir)
        else:
            # Wait for rank 0 to finish data prep
            import time
            while not (os.path.exists(train_jsonl) and os.path.exists(eval_jsonl)):
                time.sleep(5)

    # Load JSONL datasets
    data_files = {"train": train_jsonl}
    if os.path.exists(eval_jsonl):
        data_files["validation"] = eval_jsonl
    raw_ds = load_dataset("json", data_files=data_files)

    # Preprocess (create prefix_text from chat template)
    ds = raw_ds.map(make_preprocess_fn(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    if is_main:
        print(f"\n✓ Train: {len(ds['train'])} examples")
        if "validation" in ds:
            print(f"✓ Eval: {len(ds['validation'])} examples")

    # Data collator (with augmentation for train, without for eval)
    collator = DataCollatorForQwen3ASR(
        processor=processor, sampling_rate=16000,
        augment=True, spec_augment=True, speed_perturb=True,
    )
    if is_main:
        print("✓ Data augmentation: SpecAugment + Speed Perturbation")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_epochs,
        warmup_ratio=config.warmup_ratio,
        logging_steps=config.logging_steps,
        lr_scheduler_type="linear",
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=3,
        eval_strategy="steps" if "validation" in ds else "no",
        eval_steps=config.eval_steps if "validation" in ds else None,
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=["wandb"] if is_main else [],
        save_safetensors=True,
        max_grad_norm=1.0,
        weight_decay=0.01,
        dataloader_num_workers=8,
        dataloader_pin_memory=True,
        dataloader_prefetch_factor=2,
    )

    # Callbacks
    callbacks = [
        CheckpointInferableCallback(base_model_path=model_local_path),
        ComprehensiveLoggingCallback(),
    ]

    # WER/CER eval callback (rank 0 only, uses eval JSONL)
    if is_main and os.path.exists(eval_jsonl):
        wer_callback = WERCERCallback(
            processor=processor,
            eval_jsonl_path=eval_jsonl,
            wer_metric=wer_metric,
            cer_metric=cer_metric,
            num_samples=100,
        )
        callbacks.append(wer_callback)
        print(f"✓ WER/CER eval: 100 samples every {config.eval_steps} steps")

    # Trainer
    trainer = GradualUnfreezeTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        processing_class=processor.tokenizer,
        callbacks=callbacks,
    )

    # Train
    if is_main:
        print("\n" + "=" * 60)
        print("Starting training...")
        print("=" * 60)
    trainer.train()

    # Save
    if is_main:
        print("\nSaving final model...")
        trainer.save_model(config.output_dir)
        copy_required_hf_files(model_local_path, config.output_dir)

        print("\n" + "=" * 60)
        print("Training completed!")
        print(f"Model saved to: {config.output_dir}")
        print("=" * 60)

        wandb.finish()


if __name__ == "__main__":
    main()
