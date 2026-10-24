"""ASR-check clean candidate WAVs and select the clean and demo sample sets.

Run from the repository root after ``flag_wav_runs.py``:

    python experiments/clean_samples/select_clean_samples.py

Only x-vector ``clean_candidate`` rows from ``wav_flags.csv`` are eligible.
Continuation rows at NFE 20 are transcribed as a contamination check but are
never selected. WER/CER use Whisper's English text normalizer on both the
transcript and the prompt text.

Selection rule: a WAV passes when its word errors are at most
max(1, round(0.10 x reference words)). A prompt is eligible when it passes at
every selection NFE (default 20 and 32); contrast NFEs (default 4) are always
included and reported but never gate selection, so low-NFE degradation stays
visible. The clean set takes the best short and medium prompt per category by
mean CER. The demo set takes prompts with zero word errors at NFE 20 whose
selection-NFE WAVs also pass the signal checks in ``wav_flags.csv``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
FLAGS_PATH = OUT_DIR / "wav_flags.csv"
ASR_PATH = OUT_DIR / "asr_scores.csv"
CLEAN_SET_DIR = OUT_DIR / "clean_set"
DEMO_SET_DIR = OUT_DIR / "demo_set"
SAFE_PROMPTS_PATH = OUT_DIR / "safe_demo_prompts.json"
MANIFEST = ROOT / "experiments" / "benchmark_prompts_v1.csv"
BASELINE_PARTS = ("nfe_54_prompts_v1_full_part_0", "nfe_54_prompts_v1_full_part_1")
DEFAULT_ASR_MODEL = "openai/whisper-large-v3"
DEFAULT_SELECTION_NFES = (20, 32)
DEFAULT_CONTRAST_NFES = (4,)
CONTAMINATION_NFE = 20
DEMO_NFE = 20
DEMO_PROMPT_LIMIT = 8
CATEGORIES = (
    "plain",
    "numbers",
    "names_rare_words",
    "acronyms_abbreviations",
    "prosody",
    "phonetic",
)
BUCKETS = ("short", "medium")
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
        "--selection-nfes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SELECTION_NFES),
        metavar="N",
        help="NFEs a prompt must pass to be selected (default: 20 32)",
    )
    parser.add_argument(
        "--contrast-nfes",
        nargs="+",
        type=int,
        default=list(DEFAULT_CONTRAST_NFES),
        metavar="N",
        help="NFEs included and reported for every selected prompt (default: 4)",
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


def passes(row: dict[str, Any]) -> bool:
    return int(row["word_errors"]) <= allowed_word_errors(int(row["reference_words"]))


def prompt_candidates(
    scored: list[dict[str, Any]],
    flags: list[dict[str, str]],
    manifest: dict[str, dict[str, str]],
    *,
    selection_nfes: list[int],
    contrast_nfes: list[int],
) -> list[dict[str, Any]]:
    """Group clean-candidate WAVs by prompt and score prompts that qualify."""
    flag_by_path = {row["local_path"]: row for row in flags}
    wavs_by_prompt: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in scored:
        if row["status"] == "clean_candidate":
            wavs_by_prompt[row["prompt_id"]][int(row["NFE"])] = {
                **flag_by_path[row["local_path"]],
                **row,
            }

    candidates = []
    for prompt_id, wavs in sorted(wavs_by_prompt.items()):
        if any(nfe not in wavs for nfe in [*selection_nfes, *contrast_nfes]):
            continue
        selection_wavs = [wavs[nfe] for nfe in selection_nfes]
        if not all(passes(wav) for wav in selection_wavs):
            continue
        demo_wav = wavs.get(DEMO_NFE)
        candidates.append(
            {
                "prompt_id": prompt_id,
                "text": manifest[prompt_id]["text"],
                "category": manifest[prompt_id]["category"],
                "length_bucket": manifest[prompt_id]["length_bucket"],
                "mean_selection_cer": sum(float(w["CER"]) for w in selection_wavs)
                / len(selection_wavs),
                "selection_word_errors": sum(int(w["word_errors"]) for w in selection_wavs),
                "demo_safe": (
                    demo_wav is not None
                    and int(demo_wav["word_errors"]) == 0
                    and all(w["signal_ok"] == "True" for w in selection_wavs)
                ),
                "wavs": wavs,
            }
        )
    return candidates


def rank_key(candidate: dict[str, Any]) -> tuple[float, int, str]:
    return (
        candidate["mean_selection_cer"],
        candidate["selection_word_errors"],
        candidate["prompt_id"],
    )


def pick_clean_set(
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    picks, gaps = [], []
    for category in CATEGORIES:
        for bucket in BUCKETS:
            pool = [
                c
                for c in candidates
                if c["category"] == category and c["length_bucket"] == bucket
            ]
            if pool:
                picks.append(min(pool, key=rank_key))
            else:
                gaps.append(f"{category}/{bucket}")
    return picks, gaps


def pick_demo_set(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Best demo-safe prompt per category, then fill by rank up to the limit."""
    safe = sorted((c for c in candidates if c["demo_safe"]), key=rank_key)
    picks = []
    for category in CATEGORIES:
        best = next((c for c in safe if c["category"] == category), None)
        if best is not None:
            picks.append(best)
    for candidate in safe:
        if len(picks) >= DEMO_PROMPT_LIMIT:
            break
        if candidate not in picks:
            picks.append(candidate)
    return picks[:DEMO_PROMPT_LIMIT]


def sample_rows(
    picks: list[dict[str, Any]],
    nfes: list[int],
    selection_nfes: list[int],
) -> list[dict[str, Any]]:
    rows = []
    for pick in picks:
        for nfe in nfes:
            wav = pick["wavs"][nfe]
            rows.append(
                {
                    "file": f"output_{pick['prompt_id']}_nfe{nfe}.wav",
                    "prompt_id": pick["prompt_id"],
                    "text": pick["text"],
                    "category": pick["category"],
                    "length_bucket": pick["length_bucket"],
                    "NFE": nfe,
                    "role": "selection" if nfe in selection_nfes else "contrast",
                    "source_run": wav["run"],
                    "source_path": wav["local_path"],
                    "duration_seconds": float(wav["duration_seconds"]),
                    "latency_seconds": float(wav["latency_seconds"]),
                    "RTF": float(wav["RTF"]),
                    "transcript": wav["transcript"],
                    "word_errors": int(wav["word_errors"]),
                    "WER": round(float(wav["WER"]), 4),
                    "CER": round(float(wav["CER"]), 4),
                    "passes": passes(wav),
                    "signal_ok": wav["signal_ok"] == "True",
                }
            )
    return rows


def copy_samples(rows: list[dict[str, Any]], directory: Path) -> None:
    directory.mkdir(exist_ok=True)
    for stale in directory.glob("output_*.wav"):
        stale.unlink()
    for row in rows:
        shutil.copy2(ROOT / row["source_path"], directory / row["file"])


def write_manifest(
    directory: Path,
    rows: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    payload = {**metadata, "samples": rows}
    (directory / "manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    with (directory / "manifest.csv").open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_demo_readme(rows: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    lines = [
        "# Demo-safe samples",
        "",
        "Archived RTX 4090 x-vector outputs that are safe to play for an audience: no "
        f"word errors at NFE {DEMO_NFE} under `{metadata['asr_model']}`, ended before the "
        "length cap, and passed the signal checks (no clipping, no long silences, not "
        f"cut off) at NFE {', '.join(str(n) for n in metadata['selection_nfes'])}. "
        f"NFE {', '.join(str(n) for n in metadata['contrast_nfes'])} is included as a "
        "low-step contrast and may contain audible errors; its WER is reported.",
        "",
        "Fill in the **Listened** column by ear before rehearsal (Tue Oct 27). Replace a "
        "prompt that sounds wrong rather than editing the WAV.",
        "",
        "| File | Prompt | Category | NFE | Duration (s) | Latency (s) | RTF | Word errors | WER | Listened |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in rows:
        lines.append(
            f"| [{row['file']}]({row['file']}) | {row['text']} | {row['category']} | "
            f"{row['NFE']} | {row['duration_seconds']:.2f} | {row['latency_seconds']:.2f} | "
            f"{row['RTF']:.2f} | {row['word_errors']} | {row['WER']:.2f} | ☐ |"
        )
    lines.append("")
    (DEMO_SET_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    flags = load_csv(FLAGS_PATH)
    manifest = {row["id"]: row for row in load_csv(MANIFEST)}
    prompt_texts = {prompt_id: row["text"] for prompt_id, row in manifest.items()}
    nfes = sorted({*args.contrast_nfes, *args.selection_nfes})
    targets = asr_targets(flags, nfes, not args.skip_contamination_check)
    scored = transcribe(
        targets,
        prompt_texts,
        asr_model=args.asr_model,
        device=resolve_device(args.device),
    )
    write_asr_scores(scored)
    print(f"Wrote {len(scored)} rows to {ASR_PATH.relative_to(ROOT)}")

    candidates = prompt_candidates(
        scored,
        flags,
        manifest,
        selection_nfes=args.selection_nfes,
        contrast_nfes=args.contrast_nfes,
    )
    clean_picks, gaps = pick_clean_set(candidates)
    demo_picks = pick_demo_set(candidates)
    metadata = {
        "source_runs": sorted({row["run"] for row in flags if row["status"] == "clean_candidate"}),
        "asr_model": args.asr_model,
        "transformers_version": scored[0]["transformers_version"] if scored else None,
        "selection_nfes": args.selection_nfes,
        "contrast_nfes": args.contrast_nfes,
        "pass_rule": "word_errors <= max(1, round(0.10 * reference_words)) after Whisper English normalization",
        "eligible_prompts": len(candidates),
    }

    clean_rows = sample_rows(clean_picks, nfes, args.selection_nfes)
    copy_samples(clean_rows, CLEAN_SET_DIR)
    write_manifest(CLEAN_SET_DIR, clean_rows, {**metadata, "gaps": gaps})

    demo_rows = sample_rows(demo_picks, nfes, args.selection_nfes)
    copy_samples(demo_rows, DEMO_SET_DIR)
    write_manifest(
        DEMO_SET_DIR,
        demo_rows,
        {**metadata, "demo_rule": f"0 word errors at NFE {DEMO_NFE}; signal checks pass at selection NFEs"},
    )
    write_demo_readme(demo_rows, metadata)

    demo_ids = {pick["prompt_id"] for pick in demo_picks}
    safe_prompts = [
        {
            "id": c["prompt_id"],
            "text": c["text"],
            "category": c["category"],
            "length_bucket": c["length_bucket"],
            "in_demo_set": c["prompt_id"] in demo_ids,
        }
        for c in sorted((c for c in candidates if c["demo_safe"]), key=rank_key)
    ]
    SAFE_PROMPTS_PATH.write_text(
        json.dumps({**metadata, "prompts": safe_prompts}, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Eligible prompts: {len(candidates)}; demo-safe prompts: {len(safe_prompts)}")
    print(f"Clean set: {len(clean_picks)} prompts, {len(clean_rows)} WAVs; gaps: {gaps or 'none'}")
    print(f"Demo set: {len(demo_picks)} prompts, {len(demo_rows)} WAVs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
