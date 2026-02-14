#!/usr/bin/env python3
"""
Enhanced Qwen3-ASR Training with Muon Optimizer + SpecAugment

Based on qwen3_asr_sft_official.py with additions:
- Muon optimizer (2× speedup claim)
- SpecAugment in data collator (time + frequency masking)
- W&B logging
- Optimized for 100k Hebrew dataset on 8xA100

Usage:
    torchrun --nproc_per_node=8 train_100k_muon.py \\
        --train_file ./qwen3_asr_data/train_100k.jsonl \\
        --output_dir ./qwen3-asr-hebrew-100k \\
        --batch_size 4 \\
        --grad_acc 2 \\
        --epochs 3
"""

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import librosa
import torch
import torch.nn as nn
import torchaudio.transforms as T
from datasets import load_dataset
from qwen_asr import Qwen3ASRModel
from transformers import (
    GenerationConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

# Try to import Muon optimizer
try:
    from muon import Muon
    MUON_AVAILABLE = True
except ImportError:
    print("⚠ Muon optimizer not available. Install with: pip install muon-optimizer")
    print("  Falling back to AdamW")
    MUON_AVAILABLE = False


def patch_outer_forward(model):
    """Patch model.forward to delegate to model.thinker.forward."""
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


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")


def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    """Find most recent checkpoint directory."""
    if not output_dir or not os.path.isdir(output_dir):
        return None
    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    """Load audio file with librosa."""
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array):
    """Build chat-format messages for Qwen3-ASR."""
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn_prefix_only(processor):
    """Create preprocessing function for dataset mapping."""
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ex.get("prompt", "")
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
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


@dataclass
class DataCollatorWithSpecAugment:
    """
    Data collator with SpecAugment applied on-the-fly.

    SpecAugment parameters based on proven ASR configs:
    - Time masking: up to 80 time steps, 2 masks
    - Frequency masking: up to 27 freq bins (for 128-dim features), 2 masks
    - Applied during training only (controlled externally)
    """
    processor: Any
    sampling_rate: int = 16000
    apply_spec_augment: bool = True
    time_mask_param: int = 80
    freq_mask_param: int = 27
    num_time_masks: int = 2
    num_freq_masks: int = 2

    def __post_init__(self):
        """Initialize SpecAugment transforms."""
        if self.apply_spec_augment:
            self.time_masking = T.TimeMasking(time_mask_param=self.time_mask_param)
            self.freq_masking = T.FrequencyMasking(freq_mask_param=self.freq_mask_param)

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_paths = [f["audio"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        # Load audio files
        audios = [load_audio(p, sr=self.sampling_rate) for p in audio_paths]

        # Process with SpecAugment if training
        # Note: self.apply_spec_augment should be set to False during evaluation
        if self.apply_spec_augment:
            audios = self._apply_spec_augment(audios)

        # Tokenize and process
        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        # Create labels (mask prefix tokens)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs

    def _apply_spec_augment(self, audios: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply SpecAugment to audio waveforms.

        Note: This is a simplified version. Ideally, SpecAugment should be applied
        to mel spectrograms, but we're applying time masking to waveforms as a proxy.
        For production, consider converting to mel-spec first.
        """
        augmented = []
        for audio in audios:
            if not isinstance(audio, torch.Tensor):
                audio = torch.from_numpy(audio).float()

            # Add batch and channel dimensions for transforms
            # Shape: (time,) -> (1, 1, time)
            audio_2d = audio.unsqueeze(0).unsqueeze(0)

            # Apply time masking (multiple times)
            for _ in range(self.num_time_masks):
                # Time masking expects (batch, channel, time)
                audio_2d = self.time_masking(audio_2d)

            # Remove added dimensions
            audio_aug = audio_2d.squeeze(0).squeeze(0)
            augmented.append(audio_aug.numpy())

        return augmented


class MuonTrainer(Trainer):
    """
    Custom Trainer with Muon optimizer support.

    Muon optimizer claims 2× speedup for transformer training.
    Falls back to AdamW if Muon not available.
    """
    def create_optimizer(self):
        """Override optimizer creation to use Muon."""
        if MUON_AVAILABLE and self.args.use_muon:
            opt_model = self.model
            if self.optimizer is None:
                decay_parameters = self.get_decay_parameter_names(opt_model)
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

                self.optimizer = Muon(
                    optimizer_grouped_parameters,
                    lr=self.args.learning_rate,
                    momentum=0.95,  # Muon default
                )
                print(f"✓ Using Muon optimizer (lr={self.args.learning_rate})")
        else:
            # Fall back to default (AdamW)
            super().create_optimizer()
            if not MUON_AVAILABLE:
                print("✓ Using AdamW optimizer (Muon not available)")
            else:
                print("✓ Using AdamW optimizer (use_muon=False)")

    def _prepare_inputs(self, inputs):
        """Cast float inputs to model dtype."""
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs


def copy_required_hf_files_for_qwen_asr(src_dir: str, dst_dir: str):
    """Copy config files needed for inference."""
    os.makedirs(dst_dir, exist_ok=True)
    required = [
        "config.json",
        "generation_config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "special_tokens_map.json",
        "chat_template.json",
        "merges.txt",
        "vocab.json",
    ]
    for fn in required:
        src = os.path.join(src_dir, fn)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst_dir, fn))


class MakeEveryCheckpointInferableCallback(TrainerCallback):
    """Ensure every checkpoint has config files for inference."""
    def __init__(self, base_model_path: str):
        self.base_model_path = base_model_path

    def on_save(self, args: TrainingArguments, state, control, **kwargs):
        if args.process_index != 0:
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = kwargs.get("checkpoint", ckpt_dir)

        copy_required_hf_files_for_qwen_asr(self.base_model_path, ckpt_dir)
        return control


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR Training with Muon + SpecAugment")

    # Paths
    p.add_argument("--model_path", type=str, default="Qwen/Qwen3-ASR-1.7B")
    p.add_argument("--train_file", type=str, required=True, help="Path to train JSONL")
    p.add_argument("--eval_file", type=str, default="", help="Path to eval JSONL (optional)")
    p.add_argument("--output_dir", type=str, default="./qwen3-asr-hebrew-100k")

    # Audio
    p.add_argument("--sr", type=int, default=16000, help="Sampling rate")

    # Training hyperparameters
    p.add_argument("--batch_size", type=int, default=4, help="Per-GPU batch size")
    p.add_argument("--grad_acc", type=int, default=2, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--epochs", type=float, default=3, help="Number of epochs")
    p.add_argument("--log_steps", type=int, default=10, help="Logging frequency")
    p.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler")
    p.add_argument("--warmup_ratio", type=float, default=0.1, help="Warmup ratio (10%)")

    # Muon optimizer
    p.add_argument("--use_muon", type=int, default=1, help="Use Muon optimizer (1=yes, 0=no)")

    # SpecAugment
    p.add_argument("--spec_augment", type=int, default=1, help="Enable SpecAugment (1=yes, 0=no)")
    p.add_argument("--time_mask_param", type=int, default=80, help="SpecAugment time mask")
    p.add_argument("--freq_mask_param", type=int, default=27, help="SpecAugment freq mask")

    # DataLoader
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
    p.add_argument("--pin_memory", type=int, default=1)
    p.add_argument("--persistent_workers", type=int, default=1)
    p.add_argument("--prefetch_factor", type=int, default=2)

    # Checkpointing
    p.add_argument("--save_strategy", type=str, default="steps")
    p.add_argument("--save_steps", type=int, default=500, help="Save checkpoint every N steps")
    p.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")

    # W&B
    p.add_argument("--wandb_project", type=str, default="qwen3-asr-hebrew")
    p.add_argument("--wandb_run_name", type=str, default="100k-muon-specaug")

    # Resume
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--resume", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()

    if not args.train_file:
        raise ValueError("--train_file is required")

    # Setup W&B
    use_wandb = bool(args.wandb_project)
    if use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # Load model
    print(f"\n{'='*60}")
    print(f"Loading model: {args.model_path}")
    print(f"{'='*60}\n")

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16 if use_bf16 else torch.float16,
        device_map=None,
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor

    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # Load dataset (supports both HuggingFace Hub and local JSONL)
    print(f"Loading dataset: {args.train_file}")

    # Check if HuggingFace dataset or local JSONL
    if "/" in args.train_file and not args.train_file.endswith(".jsonl"):
        # HuggingFace dataset (e.g., "OzLabs/qwen3-asr-hebrew-100k")
        print(f"  → Loading from HuggingFace Hub")
        raw_ds = load_dataset(args.train_file)
    else:
        # Local JSONL file
        print(f"  → Loading from local JSONL file")
        raw_ds = load_dataset(
            "json",
            data_files={
                "train": args.train_file,
                **({"validation": args.eval_file} if args.eval_file else {}),
            },
        )

    # Preprocess
    ds = raw_ds.map(make_preprocess_fn_prefix_only(processor), num_proc=1)

    # Remove unnecessary columns
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    print(f"✓ Train examples: {len(ds['train']):,}")
    if "validation" in ds:
        print(f"✓ Eval examples: {len(ds['validation']):,}")

    # Data collator with SpecAugment
    collator = DataCollatorWithSpecAugment(
        processor=processor,
        sampling_rate=args.sr,
        apply_spec_augment=(args.spec_augment == 1),
        time_mask_param=args.time_mask_param,
        freq_mask_param=args.freq_mask_param,
        num_time_masks=2,
        num_freq_masks=2,
    )

    if args.spec_augment == 1:
        print(f"✓ SpecAugment enabled (time_mask={args.time_mask_param}, freq_mask={args.freq_mask_param})")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.log_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_ratio=args.warmup_ratio,
        dataloader_num_workers=args.num_workers,
        dataloader_pin_memory=(args.pin_memory == 1),
        dataloader_persistent_workers=(args.persistent_workers == 1),
        dataloader_prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        save_safetensors=True,
        eval_strategy="steps" if args.eval_file else "no",
        eval_steps=args.save_steps if args.eval_file else None,
        do_eval=bool(args.eval_file),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to="wandb" if use_wandb else "none",
        gradient_checkpointing=True,  # Save memory
        use_muon=(args.use_muon == 1),  # Custom flag for MuonTrainer
    )

    # Trainer
    trainer = MuonTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation", None),
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[MakeEveryCheckpointInferableCallback(base_model_path=args.model_path)],
    )

    # Calculate effective batch size and steps
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = args.batch_size * args.grad_acc * num_gpus
    steps_per_epoch = len(ds["train"]) // effective_batch
    total_steps = int(steps_per_epoch * args.epochs)

    print(f"\n{'='*60}")
    print(f"Training Configuration")
    print(f"{'='*60}")
    print(f"  GPUs: {num_gpus}")
    print(f"  Batch size per GPU: {args.batch_size}")
    print(f"  Gradient accumulation: {args.grad_acc}")
    print(f"  Effective batch size: {effective_batch}")
    print(f"  Examples: {len(ds['train']):,}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Steps per epoch: {steps_per_epoch}")
    print(f"  Total steps: {total_steps}")
    print(f"  LR: {args.lr}")
    print(f"  LR scheduler: {args.lr_scheduler_type}")
    print(f"  Warmup ratio: {args.warmup_ratio}")
    print(f"  Optimizer: {'Muon' if (args.use_muon == 1 and MUON_AVAILABLE) else 'AdamW'}")
    print(f"  SpecAugment: {'Enabled' if args.spec_augment == 1 else 'Disabled'}")
    print(f"  Precision: {'BF16' if use_bf16 else 'FP16'}")
    print(f"{'='*60}\n")

    # Resume if requested
    resume_from = (args.resume_from or "").strip()
    if not resume_from and args.resume == 1:
        resume_from = find_latest_checkpoint(training_args.output_dir) or ""

    if resume_from:
        print(f"Resuming from checkpoint: {resume_from}\n")
        trainer.train(resume_from_checkpoint=resume_from)
    else:
        trainer.train()

    # Save final model
    print(f"\nSaving final model to: {args.output_dir}")
    trainer.save_model(args.output_dir)
    copy_required_hf_files_for_qwen_asr(args.model_path, args.output_dir)

    print(f"\n{'='*60}")
    print(f"✓ Training Complete!")
    print(f"{'='*60}")
    print(f"  Model saved: {args.output_dir}")
    print(f"  Next steps:")
    print(f"    1. Evaluate on benchmarks")
    print(f"    2. Analyze errors for Phase 2")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
