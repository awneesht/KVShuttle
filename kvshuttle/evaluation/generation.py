"""Generation-based evaluation: GSM8K and MMLU subsets."""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def extract_gsm8k_answer(text: str) -> str | None:
    """Extract numerical answer from GSM8K generation.

    Looks for the pattern #### <number> or the last number in the text.
    """
    # Look for #### pattern
    match = re.search(r"####\s*([\d,]+(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")

    # Fallback: last number in the text
    numbers = re.findall(r"[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def evaluate_gsm8k_accuracy(
    predictions: list[str],
    references: list[str],
) -> float:
    """Compute GSM8K accuracy by comparing extracted numerical answers.

    Args:
        predictions: Generated text answers.
        references: Ground truth answers (from GSM8K "answer" field).

    Returns:
        Accuracy in [0, 1].
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, references):
        pred_answer = extract_gsm8k_answer(pred)
        ref_answer = extract_gsm8k_answer(ref)

        if pred_answer is not None and ref_answer is not None:
            try:
                if float(pred_answer) == float(ref_answer):
                    correct += 1
            except ValueError:
                if pred_answer.strip() == ref_answer.strip():
                    correct += 1

    return correct / total if total > 0 else 0.0


def evaluate_mmlu_accuracy(
    predictions: list[str],
    correct_answers: list[str],
) -> float:
    """Compute MMLU accuracy from multiple-choice predictions.

    Args:
        predictions: Model predictions (should be A/B/C/D).
        correct_answers: Ground truth answers.

    Returns:
        Accuracy in [0, 1].
    """
    correct = 0
    total = len(predictions)

    for pred, ref in zip(predictions, correct_answers):
        # Extract first letter that is A-D
        pred_letter = _extract_choice(pred)
        ref_letter = _extract_choice(ref)
        if pred_letter and ref_letter and pred_letter == ref_letter:
            correct += 1

    return correct / total if total > 0 else 0.0


def _extract_choice(text: str) -> str | None:
    """Extract A/B/C/D choice from text."""
    match = re.search(r"\b([A-D])\b", text.upper())
    return match.group(1) if match else None
