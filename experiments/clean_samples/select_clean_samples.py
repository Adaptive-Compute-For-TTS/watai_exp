"""ASR-check clean candidate WAVs and select the clean and demo sample sets.

Run from the repository root after ``flag_wav_runs.py``:

    python experiments/clean_samples/select_clean_samples.py

Only x-vector ``clean_candidate`` rows from ``wav_flags.csv`` are eligible.
Continuation rows at NFE 20 are transcribed as a contamination check but are
never selected. WER/CER use Whisper's English text normalizer on both the
transcript and the prompt text.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FLAGS_PATH = OUT_DIR / "wav_flags.csv"
ASR_PATH = OUT_DIR / "asr_scores.csv"
MANIFEST = ROOT / "experiments" / "benchmark_prompts_v1.csv"
BASELINE_PARTS = ("nfe_54_prompts_v1_full_part_0", "nfe_54_prompts_v1_full_part_1")
DEFAULT_ASR_MODEL = "openai/whisper-large-v3"
DEFAULT_NFES = (4, 20, 32)
CONTAMINATION_NFE = 20
ASR_SAMPLE_RATE = 16000
REFERENCE_PHRASES = ("adam here", "feels real")
ASR_FIELDS = (
    "local_path",
    "run",
    "prompt_id",
    "NFE",
    "conditioning",
    "status",
    "prompt_text",
    "transcript",
    "normalized_reference",
    "normalized_transcript",
    "reference_words",
    "word_errors",
    "WER",
    "CER",
    "mentions_reference_phrase",
    "asr_model",
    "transformers_version",
)


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--asr-model",
        default=DEFAULT_ASR_MODEL,
        help=f"Hugging Face Whisper model ID (default: {DEFAULT_ASR_MODEL})",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for ASR: auto, mps, cuda, or cpu (default: auto)",
    )
    parser.add_argument(
        "--nfe-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_NFES),
        metavar="N",
        help="NFE values for the clean set (default: 4 20 32)",
    )
    parser.add_argument(
        "--skip-contamination-check",
        action="store_true",
        help=f"Do not transcribe continuation-run WAVs at NFE {CONTAMINATION_NFE}",
    )
    return parser.parse_args(argv)


def asr_targets(
    flags: list[dict[str, str]],
    nfe_values: list[int],
    include_contamination: bool,
) -> list[dict[str, str]]:
    targets = [
        row
        for row in flags
        if row["status"] == "clean_candidate" and int(row["NFE"]) in nfe_values
    ]
    if include_contamination:
        targets += [
            row
            for row in flags
            if row["run"] in BASELINE_PARTS and int(row["NFE"]) == CONTAMINATION_NFE
        ]
    return targets


def resolve_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe(
    targets: list[dict[str, str]],
    prompt_texts: dict[str, str],
    *,
    asr_model: str,
    device: str,
) -> list[dict[str, Any]]:
    """Transcribe targets, reusing rows already scored with the same model."""
    import librosa
    import transformers
    from transformers import WhisperProcessor, pipeline

    cached = {}
    if ASR_PATH.is_file():
        cached = {
            row["local_path"]: row
            for row in load_csv(ASR_PATH)
            if row["asr_model"] == asr_model
        }
    pending = [row for row in targets if row["local_path"] not in cached]
    print(
        f"ASR {asr_model} on {device}: {len(targets)} targets, "
        f"{len(targets) - len(pending)} cached, {len(pending)} to transcribe"
    )

    normalize = WhisperProcessor.from_pretrained(asr_model).tokenizer.normalize
    recognizer = None
    if pending:
        recognizer = pipeline(
            "automatic-speech-recognition",
            model=asr_model,
            device=device,
        )

    scored: list[dict[str, Any]] = []
    started = time.perf_counter()
    for index, row in enumerate(targets, start=1):
        if row["local_path"] in cached:
            transcript = cached[row["local_path"]]["transcript"]
        else:
            audio, _ = librosa.load(ROOT / row["local_path"], sr=ASR_SAMPLE_RATE, mono=True)
            output = recognizer(
                {"raw": audio, "sampling_rate": ASR_SAMPLE_RATE},
                generate_kwargs={"language": "english", "task": "transcribe", "num_beams": 1},
            )
            transcript = output["text"].strip()
            if index % 10 == 0 or index == len(targets):
                print(f"  {index}/{len(targets)} transcribed ({time.perf_counter() - started:.0f}s)")
        prompt_text = prompt_texts[row["prompt_id"]]
        normalized_reference = normalize(prompt_text)
        normalized_transcript = normalize(transcript)
        scored.append(
            {
                "local_path": row["local_path"],
                "run": row["run"],
                "prompt_id": row["prompt_id"],
                "NFE": int(row["NFE"]),
                "conditioning": row["conditioning"],
                "status": row["status"],
                "prompt_text": prompt_text,
                "transcript": transcript,
                "normalized_reference": normalized_reference,
                "normalized_transcript": normalized_transcript,
                **error_scores(normalized_reference, normalized_transcript),
                "mentions_reference_phrase": mentions_reference_phrase(normalized_transcript),
                "asr_model": asr_model,
                "transformers_version": transformers.__version__,
            }
        )
    return scored


def write_asr_scores(scored: list[dict[str, Any]]) -> None:
    with ASR_PATH.open("w", newline="", encoding="utf-8") as asr_file:
        writer = csv.DictWriter(asr_file, fieldnames=ASR_FIELDS)
        writer.writeheader()
        for row in scored:
            writer.writerow(
                {
                    **row,
                    "WER": f"{row['WER']:.4f}",
                    "CER": f"{row['CER']:.4f}",
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    flags = load_csv(FLAGS_PATH)
    prompt_texts = {row["id"]: row["text"] for row in load_csv(MANIFEST)}
    targets = asr_targets(flags, args.nfe_values, not args.skip_contamination_check)
    scored = transcribe(
        targets,
        prompt_texts,
        asr_model=args.asr_model,
        device=resolve_device(args.device),
    )
    write_asr_scores(scored)
    print(f"Wrote {len(scored)} rows to {ASR_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
