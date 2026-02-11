#!/usr/bin/env python3
"""
Phase 0: Stratified Forced Aligner Quality Audit

Samples 10% of training data (stratified by domain) and runs Qwen3-ForcedAligner
to assess data quality BEFORE spending $88 on training.

Decision Gate:
  - If >15% have coverage <0.6 → STOP, do filtering/resegmentation first
  - If <10% have coverage <0.6 → PROCEED to Phase 2 training

Usage:
    uv run python scripts/phase0_align_audit.py
"""

import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm

# Optional wandb integration
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class AlignmentMetrics:
    """Per-sample alignment quality metrics."""
    sample_id: str
    domain: str
    audio_duration: float
    transcript_length: int
    alignment_coverage: float
    alignment_confidence: float
    length_mismatch: bool
    low_quality: bool


class Qwen3AlignerAuditor:
    """
    Quality auditor using Qwen3-ForcedAligner.

    Performs stratified sampling and computes alignment quality metrics
    to determine if data needs filtering before training.
    """

    def __init__(
        self,
        sample_rate: float = 0.10,  # 10% sample
        coverage_threshold: float = 0.6,  # Low quality if <0.6
        confidence_threshold: float = 0.5,
    ):
        self.sample_rate = sample_rate
        self.coverage_threshold = coverage_threshold
        self.confidence_threshold = confidence_threshold

        # Try to load aligner
        self.aligner = None
        try:
            from qwen_asr import Qwen3ForcedAligner
            print("Loading Qwen3-ForcedAligner-0.6B...")
            self.aligner = Qwen3ForcedAligner.from_pretrained(
                "Qwen/Qwen3-ForcedAligner-0.6B"
            )
            print("✓ Forced aligner loaded")
        except ImportError:
            print("⚠️  qwen-asr not installed. Install with: pip install qwen-asr")
            print("⚠️  Falling back to heuristic quality checks")
        except Exception as e:
            print(f"⚠️  Failed to load aligner: {e}")
            print("⚠️  Falling back to heuristic quality checks")

    def categorize_sample_domain(
        self,
        dataset_name: str,
        audio_duration: float,
        text: str
    ) -> str:
        """
        Categorize sample into domain for stratified sampling.

        Returns:
            "whatsapp_noisy", "kan_clean", or "long_tail"
        """
        # WhatsApp/noisy indicators
        if "whatsapp" in dataset_name.lower():
            return "whatsapp_noisy"

        # KAN/broadcast/clean indicators
        if any(x in dataset_name.lower() for x in ["kan", "saspeech", "broadcast"]):
            return "kan_clean"

        # Long tail: long segments or specific patterns
        if audio_duration > 20.0:
            return "long_tail"

        # Default based on characteristics
        # Short, clean recordings → likely clean
        if audio_duration < 10.0 and len(text) > 30:
            return "kan_clean"

        # Default to noisy
        return "whatsapp_noisy"

    def compute_alignment_metrics(
        self,
        audio_path: str,
        transcript: str,
        sample_id: str,
        domain: str
    ) -> AlignmentMetrics:
        """
        Compute alignment quality metrics for a single sample.

        Uses forced aligner if available, otherwise falls back to heuristics.
        """
        import soundfile as sf

        # Load audio to get duration
        audio, sr = sf.read(audio_path)
        audio_duration = len(audio) / sr
        transcript_length = len(transcript)

        if self.aligner is not None:
            # Use forced aligner
            try:
                result = self.aligner.align(
                    audio=audio_path,
                    text=transcript,
                    language="Hebrew"
                )

                if result and 'words' in result:
                    # Compute coverage
                    aligned_words = len(result['words'])
                    expected_words = len(transcript.split())
                    coverage = aligned_words / max(expected_words, 1)

                    # Compute confidence (average of word confidences if available)
                    # Note: Qwen3-ForcedAligner may not provide per-word confidence
                    # so we estimate from alignment quality
                    confidence = min(coverage, 1.0)

                else:
                    # Alignment failed
                    coverage = 0.0
                    confidence = 0.0

            except Exception as e:
                print(f"  Warning: Alignment failed for {sample_id}: {e}")
                coverage = 0.0
                confidence = 0.0

        else:
            # Heuristic quality check based on chars/second
            chars_per_sec = transcript_length / max(audio_duration, 0.1)

            # Hebrew typical: 8-15 chars/sec
            if 5.0 <= chars_per_sec <= 20.0:
                coverage = 0.8  # Assume good
                confidence = 0.7
            elif 3.0 <= chars_per_sec <= 25.0:
                coverage = 0.6  # Assume medium
                confidence = 0.5
            else:
                coverage = 0.3  # Assume poor
                confidence = 0.3

        # Check length mismatch
        expected_duration = transcript_length / 10.0  # Rough estimate: 10 chars/sec
        length_mismatch = abs(audio_duration - expected_duration) > max(audio_duration, expected_duration) * 0.5

        # Determine if low quality
        low_quality = (
            coverage < self.coverage_threshold or
            confidence < self.confidence_threshold or
            length_mismatch
        )

        return AlignmentMetrics(
            sample_id=sample_id,
            domain=domain,
            audio_duration=audio_duration,
            transcript_length=transcript_length,
            alignment_coverage=coverage,
            alignment_confidence=confidence,
            length_mismatch=length_mismatch,
            low_quality=low_quality
        )

    def stratified_sample_datasets(
        self,
        datasets_config: List[Tuple[str, str]]
    ) -> Dict[str, List]:
        """
        Load datasets and perform stratified sampling.

        Args:
            datasets_config: List of (dataset_name, split) tuples

        Returns:
            Dict mapping domain to list of samples
        """
        print("\n" + "="*70)
        print("Loading and Categorizing Datasets")
        print("="*70)

        # Collect all samples with domain labels
        all_samples = []

        for dataset_name, split in datasets_config:
            print(f"\nLoading {dataset_name} ({split})...")
            try:
                ds = load_dataset(dataset_name, split=split)
                ds = ds.cast_column("audio", Audio(sampling_rate=16000))

                # Categorize each sample
                for i, example in enumerate(tqdm(ds, desc="Categorizing")):
                    audio_data = example["audio"]
                    text = example.get("text", example.get("transcript", ""))

                    # Get duration
                    audio_array = audio_data["array"]
                    duration = len(audio_array) / audio_data["sampling_rate"]

                    domain = self.categorize_sample_domain(dataset_name, duration, text)

                    all_samples.append({
                        "dataset": dataset_name,
                        "index": i,
                        "domain": domain,
                        "audio": audio_data,
                        "text": text,
                        "duration": duration
                    })

            except Exception as e:
                print(f"✗ Failed to load {dataset_name}: {e}")
                continue

        # Group by domain
        domain_samples = defaultdict(list)
        for sample in all_samples:
            domain_samples[sample["domain"]].append(sample)

        print(f"\n✓ Total samples: {len(all_samples)}")
        for domain, samples in domain_samples.items():
            print(f"  {domain}: {len(samples)} samples")

        # Stratified sampling: 40% WhatsApp, 40% KAN, 20% long-tail
        target_distribution = {
            "whatsapp_noisy": 0.40,
            "kan_clean": 0.40,
            "long_tail": 0.20
        }

        total_sample_size = int(len(all_samples) * self.sample_rate)
        stratified_samples = {}

        print(f"\nStratified sampling ({total_sample_size} samples, {self.sample_rate*100}%):")
        for domain, ratio in target_distribution.items():
            domain_sample_size = int(total_sample_size * ratio)
            available = len(domain_samples[domain])

            if available < domain_sample_size:
                print(f"  {domain}: {available} (requested {domain_sample_size}, limited by availability)")
                stratified_samples[domain] = domain_samples[domain]
            else:
                # Random sample
                indices = np.random.choice(available, domain_sample_size, replace=False)
                stratified_samples[domain] = [domain_samples[domain][i] for i in indices]
                print(f"  {domain}: {domain_sample_size} samples")

        return stratified_samples

    def audit_samples(
        self,
        stratified_samples: Dict[str, List],
        output_dir: str = "./phase0_audit_results"
    ) -> Dict:
        """
        Run forced aligner audit on stratified samples.

        Returns:
            Audit report with statistics and decision gate recommendation
        """
        os.makedirs(output_dir, exist_ok=True)

        print("\n" + "="*70)
        print("Running Forced Aligner Quality Audit")
        print("="*70)

        all_metrics = []

        for domain, samples in stratified_samples.items():
            print(f"\nProcessing {domain} ({len(samples)} samples)...")

            for sample in tqdm(samples, desc=domain):
                try:
                    # Save audio temporarily (sanitize dataset name for filesystem)
                    safe_dataset = sample['dataset'].replace('/', '_')
                    temp_audio = Path(output_dir) / f"temp_{safe_dataset}_{sample['index']}.wav"

                    import soundfile as sf
                    sf.write(
                        str(temp_audio),
                        sample["audio"]["array"],
                        sample["audio"]["sampling_rate"]
                    )

                    # Compute metrics
                    metrics = self.compute_alignment_metrics(
                        audio_path=str(temp_audio),
                        transcript=sample["text"],
                        sample_id=f"{sample['dataset']}_{sample['index']}",
                        domain=domain
                    )

                    all_metrics.append(metrics)

                    # Clean up temp file
                    temp_audio.unlink()
                except Exception as e:
                    print(f"\n  Warning: skipping sample {sample['dataset']}_{sample['index']}: {e}")
                    if temp_audio.exists():
                        temp_audio.unlink()

        # Compute statistics
        report = self._generate_report(all_metrics, output_dir)

        return report

    def _generate_report(
        self,
        all_metrics: List[AlignmentMetrics],
        output_dir: str
    ) -> Dict:
        """Generate comprehensive quality report with decision gate."""

        # Overall statistics
        total = len(all_metrics)
        low_quality_count = sum(1 for m in all_metrics if m.low_quality)
        low_quality_pct = (low_quality_count / total) * 100 if total > 0 else 0

        # Per-domain statistics
        domain_stats = {}
        for domain in ["whatsapp_noisy", "kan_clean", "long_tail"]:
            domain_metrics = [m for m in all_metrics if m.domain == domain]
            if domain_metrics:
                domain_stats[domain] = {
                    "count": len(domain_metrics),
                    "low_quality_count": sum(1 for m in domain_metrics if m.low_quality),
                    "low_quality_pct": (sum(1 for m in domain_metrics if m.low_quality) / len(domain_metrics)) * 100,
                    "mean_coverage": np.mean([m.alignment_coverage for m in domain_metrics]),
                    "median_coverage": np.median([m.alignment_coverage for m in domain_metrics]),
                    "mean_confidence": np.mean([m.alignment_confidence for m in domain_metrics]),
                    "length_mismatch_pct": (sum(1 for m in domain_metrics if m.length_mismatch) / len(domain_metrics)) * 100,
                }

        # Coverage distribution
        coverages = [m.alignment_coverage for m in all_metrics]
        coverage_dist = {
            "p10": np.percentile(coverages, 10),
            "p25": np.percentile(coverages, 25),
            "p50": np.percentile(coverages, 50),
            "p75": np.percentile(coverages, 75),
            "p90": np.percentile(coverages, 90),
            "mean": np.mean(coverages),
        }

        # Decision gate
        if low_quality_pct > 15.0:
            decision = "STOP"
            recommendation = (
                f"❌ STOP: {low_quality_pct:.1f}% of samples have low quality (>15% threshold). "
                "Filtering and resegmentation should be done BEFORE training. "
                "Expected ROI from data cleaning is higher than training improvements."
            )
        elif low_quality_pct < 10.0:
            decision = "PROCEED"
            recommendation = (
                f"✅ PROCEED: Only {low_quality_pct:.1f}% of samples have low quality (<10% threshold). "
                "Data quality is acceptable for Round 2 training. "
                "Expected WER improvement from training: 1-2 points."
            )
        else:
            decision = "CAUTION"
            recommendation = (
                f"⚠️  CAUTION: {low_quality_pct:.1f}% of samples have low quality (10-15% range). "
                "Training may proceed but consider light filtering to remove worst 5-10%. "
                "Monitor validation loss carefully during training."
            )

        report = {
            "total_samples": total,
            "low_quality_count": low_quality_count,
            "low_quality_percentage": low_quality_pct,
            "decision": decision,
            "recommendation": recommendation,
            "coverage_distribution": coverage_dist,
            "domain_statistics": domain_stats,
            "detailed_metrics": [asdict(m) for m in all_metrics]
        }

        # Save report
        report_path = Path(output_dir) / "alignment_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        print("\n" + "="*70)
        print("PHASE 0 AUDIT RESULTS")
        print("="*70)
        print(f"\nTotal samples audited: {total}")
        print(f"Low quality samples: {low_quality_count} ({low_quality_pct:.1f}%)")
        print(f"\nCoverage distribution:")
        for k, v in coverage_dist.items():
            print(f"  {k}: {v:.3f}")
        print(f"\nPer-domain breakdown:")
        for domain, stats in domain_stats.items():
            print(f"\n  {domain}:")
            print(f"    Samples: {stats['count']}")
            print(f"    Low quality: {stats['low_quality_pct']:.1f}%")
            print(f"    Mean coverage: {stats['mean_coverage']:.3f}")
            print(f"    Length mismatches: {stats['length_mismatch_pct']:.1f}%")
        print(f"\n{'='*70}")
        print("DECISION GATE")
        print("="*70)
        print(f"\n{recommendation}")
        print(f"\n✓ Full report saved to: {report_path}")
        print("="*70)

        # Log to wandb if available and enabled
        if WANDB_AVAILABLE and os.environ.get("WANDB_PHASE0_LOGGING", "false").lower() == "true":
            try:
                wandb.init(
                    project=os.environ.get("WANDB_PROJECT", "qwen3-asr-hebrew"),
                    name="phase0-audit",
                    job_type="data_quality",
                    tags=["phase0", "data-audit", "forced-aligner"],
                )

                # Log summary metrics
                wandb.log({
                    "phase0/total_samples": total,
                    "phase0/low_quality_count": low_quality_count,
                    "phase0/low_quality_percentage": low_quality_pct,
                    "phase0/decision": decision,
                    "phase0/coverage_p10": coverage_dist["p10"],
                    "phase0/coverage_p25": coverage_dist["p25"],
                    "phase0/coverage_median": coverage_dist["p50"],
                    "phase0/coverage_p75": coverage_dist["p75"],
                    "phase0/coverage_p90": coverage_dist["p90"],
                    "phase0/coverage_mean": coverage_dist["mean"],
                })

                # Log per-domain metrics
                for domain, stats in domain_stats.items():
                    wandb.log({
                        f"phase0/{domain}/count": stats["count"],
                        f"phase0/{domain}/low_quality_pct": stats["low_quality_pct"],
                        f"phase0/{domain}/mean_coverage": stats["mean_coverage"],
                        f"phase0/{domain}/median_coverage": stats["median_coverage"],
                        f"phase0/{domain}/mean_confidence": stats["mean_confidence"],
                        f"phase0/{domain}/length_mismatch_pct": stats["length_mismatch_pct"],
                    })

                # Save report as artifact
                artifact = wandb.Artifact("phase0-audit-report", type="report")
                artifact.add_file(str(report_path))
                wandb.log_artifact(artifact)

                wandb.finish()
                print("✓ Phase 0 results logged to Weights & Biases")
            except Exception as e:
                print(f"⚠️  Failed to log to wandb: {e}")

        return report


def main():
    """Run Phase 0 forced aligner quality audit."""

    # Datasets to audit (from your existing training setup)
    datasets_config = [
        ("ivrit-ai/crowd-transcribe-v5", "train"),
        ("ivrit-ai/crowd-recital-whisper-training", "train"),
    ]

    # Initialize auditor
    auditor = Qwen3AlignerAuditor(
        sample_rate=0.10,  # 10% sample
        coverage_threshold=0.6,
        confidence_threshold=0.5
    )

    # Perform stratified sampling
    stratified_samples = auditor.stratified_sample_datasets(datasets_config)

    # Run audit
    report = auditor.audit_samples(
        stratified_samples,
        output_dir="./phase0_audit_results"
    )

    # Exit with code based on decision
    if report["decision"] == "STOP":
        sys.exit(1)  # Non-zero exit to signal stopping
    else:
        sys.exit(0)  # Success


if __name__ == "__main__":
    main()
