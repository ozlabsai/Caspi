"""Metrics calculation for ASR evaluation."""

from typing import List

from jiwer import wer as compute_wer


def calculate_wer(references: List[str], hypotheses: List[str]) -> float:
    """
    Calculate Word Error Rate (WER).

    Args:
        references: List of reference transcriptions
        hypotheses: List of hypothesis transcriptions

    Returns:
        WER score (0.0 to 1.0+)
    """
    if not references or not hypotheses:
        raise ValueError("References and hypotheses must not be empty")

    if len(references) != len(hypotheses):
        raise ValueError("References and hypotheses must have the same length")

    return compute_wer(references, hypotheses)
