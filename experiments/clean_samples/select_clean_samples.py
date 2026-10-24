"""ASR-check clean candidate WAVs and select the clean and demo sample sets.

Run from the repository root after ``flag_wav_runs.py``:

    python experiments/clean_samples/select_clean_samples.py

Only x-vector ``clean_candidate`` rows from ``wav_flags.csv`` are eligible.
Continuation rows at NFE 20 are transcribed as a contamination check but are
never selected. WER/CER use Whisper's English text normalizer on both the
transcript and the prompt text.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FLAGS_PATH = OUT_DIR / "wav_flags.csv"
MANIFEST = ROOT / "experiments" / "benchmark_prompts_v1.csv"
REFERENCE_PHRASES = ("adam here", "feels real")


def edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ref_item != hyp_item),
                )
            )
        previous = current
    return previous[-1]


def error_scores(reference: str, hypothesis: str) -> dict[str, float | int]:
    """Return word errors, WER, and CER for already normalized text."""
    ref_words, hyp_words = reference.split(), hypothesis.split()
    ref_chars, hyp_chars = " ".join(ref_words), " ".join(hyp_words)
    word_errors = edit_distance(ref_words, hyp_words)
    char_errors = edit_distance(ref_chars, hyp_chars)
    return {
        "reference_words": len(ref_words),
        "word_errors": word_errors,
        "WER": word_errors / max(len(ref_words), 1),
        "CER": char_errors / max(len(ref_chars), 1),
    }


def mentions_reference_phrase(normalized_transcript: str) -> bool:
    return any(phrase in normalized_transcript for phrase in REFERENCE_PHRASES)


def allowed_word_errors(reference_words: int) -> int:
    return max(1, round(0.10 * reference_words))


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))
