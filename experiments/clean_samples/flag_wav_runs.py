"""Flag every archived NFE benchmark WAV as clean, truncated, or contaminated.

Run from the repository root:

    python experiments/clean_samples/flag_wav_runs.py

Two problems affect the archived runs:

* Truncation. Every run used ``max_generate_length=80``. One latent patch is
  0.16 s of 48 kHz audio, so x-vector output is capped at 12.8 s; a WAV at the
  cap stopped because it ran out of budget, not because the model chose to end.
* Reference-intro contamination. Runs without ``voice_conditioning: x-vector``
  in ``run_metadata.json`` predate commit 65d27f5, when the benchmark always
  passed the reference transcript (continuation mode). Those outputs are not
  usable as clean samples.

``results.csv`` stores cluster paths, so each WAV is found by file name inside
the run folder that holds that ``results.csv``.

Writes ``wav_flags.csv`` (one row per WAV, including signal checks) and
``run_summary.json`` (per-NFE medians and cap counts for the two full runs).
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "experiments" / "nfe_benchmark_runs"
MANIFEST = ROOT / "experiments" / "benchmark_prompts_v1.csv"
OUT_DIR = Path(__file__).resolve().parent
FLAGS_PATH = OUT_DIR / "wav_flags.csv"
SUMMARY_PATH = OUT_DIR / "run_summary.json"

# patch_size 4 x vocoder hop 1920 = 7680 samples = 0.16 s at 48 kHz.
PATCH_SECONDS = 0.16
CAP_TOLERANCE_SECONDS = 0.01
FRAME_SECONDS = 0.02
SILENCE_BELOW_PEAK_DB = 40.0

BASELINE_PARTS = ("nfe_54_prompts_v1_full_part_0", "nfe_54_prompts_v1_full_part_1")
XVECTOR_PARTS = (
    "nfe_54_prompts_v1_full_xvector_part_0",
    "nfe_54_prompts_v1_full_xvector_part_1",
)

FLAG_FIELDS = (
    "run",
    "local_path",
    "prompt_id",
    "NFE",
    "repeat",
    "conditioning",
    "length_bucket",
    "category",
    "duration_seconds",
    "latency_seconds",
    "RTF",
    "max_generate_length",
    "cap_seconds",
    "at_cap",
    "peak",
    "clipped_samples",
    "leading_silence_seconds",
    "trailing_silence_seconds",
    "longest_internal_gap_seconds",
    "signal_ok",
    "status",
    "reason",
)


def load_manifest() -> dict[str, dict[str, str]]:
    with MANIFEST.open(newline="", encoding="utf-8") as manifest_file:
        return {row["id"]: row for row in csv.DictReader(manifest_file)}


def run_conditioning(run_dir: Path) -> str:
    metadata_path = run_dir / "run_metadata.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("voice_conditioning") == "x-vector":
            return "x-vector"
    return "continuation"


def local_wav_path(run_dir: Path, recorded_path: str) -> Path:
    path = run_dir / Path(recorded_path).name
    if not path.is_file():
        raise FileNotFoundError(f"WAV listed in {run_dir.name}/results.csv is missing: {path}")
    return path


def signal_checks(path: Path) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    frame = int(FRAME_SECONDS * sample_rate)
    frame_count = len(audio) // frame
    frames = audio[: frame_count * frame].reshape(frame_count, frame)
    level_db = 20 * np.log10(np.sqrt((frames**2).mean(axis=1)) + 1e-9)
    voiced = np.flatnonzero(level_db > level_db.max() - SILENCE_BELOW_PEAK_DB)

    longest_gap = 0
    if voiced.size > 1:
        longest_gap = int((np.diff(voiced) - 1).max())
    return {
        "duration_seconds": len(audio) / sample_rate,
        "peak": float(np.abs(audio).max()),
        "clipped_samples": int((np.abs(audio) >= 0.999).sum()),
        "leading_silence_seconds": voiced[0] * FRAME_SECONDS if voiced.size else 0.0,
        "trailing_silence_seconds": (
            (frame_count - 1 - voiced[-1]) * FRAME_SECONDS if voiced.size else 0.0
        ),
        "longest_internal_gap_seconds": longest_gap * FRAME_SECONDS,
    }


def signal_ok(checks: dict[str, Any]) -> bool:
    return (
        checks["clipped_samples"] == 0
        and checks["peak"] <= 0.95
        and checks["leading_silence_seconds"] < 0.5
        and checks["trailing_silence_seconds"] >= 0.05
        and checks["longest_internal_gap_seconds"] <= 1.0
    )


def classify(conditioning: str, at_cap: bool, max_generate_length: int) -> tuple[str, str]:
    if conditioning == "continuation":
        return (
            "contaminated_continuation",
            "Reference transcript was passed (continuation mode); output may "
            "contain the reference intro and is not a clean sample.",
        )
    if at_cap:
        return (
            "truncated_at_cap",
            f"Stopped at max_generate_length={max_generate_length} "
            f"({max_generate_length * PATCH_SECONDS:.2f} s), likely mid-sentence.",
        )
    return "clean_candidate", "X-vector conditioning and ended before the length cap."


def flag_rows(manifest: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    flagged: list[dict[str, Any]] = []
    for results_path in sorted(RUN_ROOT.glob("*/results.csv")):
        run_dir = results_path.parent
        conditioning = run_conditioning(run_dir)
        with results_path.open(newline="", encoding="utf-8") as results_file:
            for row in csv.DictReader(results_file):
                wav = local_wav_path(run_dir, row["output_path"])
                checks = signal_checks(wav)
                max_generate_length = int(row["max_generate_length"])
                cap_seconds = max_generate_length * PATCH_SECONDS
                at_cap = checks["duration_seconds"] >= cap_seconds - CAP_TOLERANCE_SECONDS
                status, reason = classify(conditioning, at_cap, max_generate_length)
                prompt = manifest.get(row["prompt_id"], {})
                flagged.append(
                    {
                        "run": run_dir.name,
                        "local_path": str(wav.relative_to(ROOT)),
                        "prompt_id": row["prompt_id"],
                        "NFE": int(row["NFE"]),
                        "repeat": int(row["repeat"]),
                        "conditioning": conditioning,
                        "length_bucket": prompt.get("length_bucket", ""),
                        "category": prompt.get("category", ""),
                        "duration_seconds": round(checks["duration_seconds"], 4),
                        "latency_seconds": float(row["latency_seconds"]),
                        "RTF": float(row["RTF"]),
                        "max_generate_length": max_generate_length,
                        "cap_seconds": round(cap_seconds, 2),
                        "at_cap": at_cap,
                        "peak": round(checks["peak"], 4),
                        "clipped_samples": checks["clipped_samples"],
                        "leading_silence_seconds": round(checks["leading_silence_seconds"], 2),
                        "trailing_silence_seconds": round(checks["trailing_silence_seconds"], 2),
                        "longest_internal_gap_seconds": round(
                            checks["longest_internal_gap_seconds"], 2
                        ),
                        "signal_ok": signal_ok(checks),
                        "status": status,
                        "reason": reason,
                    }
                )
    return flagged


def median(values: list[float]) -> float:
    return statistics.median(values) if values else float("nan")


def run_group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_nfe: dict[int, list[dict[str, Any]]] = defaultdict(list)
    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_nfe[row["NFE"]].append(row)
        by_bucket[row["length_bucket"]].append(row)
    max_duration = max(row["duration_seconds"] for row in rows)
    return {
        "rows": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "max_duration_seconds": max_duration,
        "rows_at_run_max_duration": sum(
            1 for row in rows if row["duration_seconds"] >= max_duration - CAP_TOLERANCE_SECONDS
        ),
        "per_nfe": {
            str(nfe): {
                "latency_median_s": median([r["latency_seconds"] for r in group]),
                "duration_median_s": median([r["duration_seconds"] for r in group]),
                "rtf_median": median([r["RTF"] for r in group]),
            }
            for nfe, group in sorted(by_nfe.items())
        },
        "per_bucket": {
            bucket: {
                "rows": len(group),
                "at_cap": sum(1 for r in group if r["at_cap"]),
                "at_run_max_duration": sum(
                    1
                    for r in group
                    if r["duration_seconds"] >= max_duration - CAP_TOLERANCE_SECONDS
                ),
                "latency_median_s": median([r["latency_seconds"] for r in group]),
                "duration_median_s": median([r["duration_seconds"] for r in group]),
                "rtf_median": median([r["RTF"] for r in group]),
            }
            for bucket, group in sorted(by_bucket.items())
        },
    }


def build_summary(flagged: list[dict[str, Any]]) -> dict[str, Any]:
    baseline = [row for row in flagged if row["run"] in BASELINE_PARTS]
    xvector = [row for row in flagged if row["run"] in XVECTOR_PARTS]
    baseline_summary = run_group_summary(baseline)
    xvector_summary = run_group_summary(xvector)
    per_nfe_change = {}
    for nfe, base in baseline_summary["per_nfe"].items():
        xvec = xvector_summary["per_nfe"][nfe]
        per_nfe_change[nfe] = {
            "latency_median_change_pct": 100 * (xvec["latency_median_s"] / base["latency_median_s"] - 1),
            "rtf_median_change_pct": 100 * (xvec["rtf_median"] / base["rtf_median"] - 1),
            "duration_median_change_s": xvec["duration_median_s"] - base["duration_median_s"],
        }
    return {
        "patch_seconds": PATCH_SECONDS,
        "all_runs_status_counts": dict(Counter(row["status"] for row in flagged)),
        "rows_per_run": dict(Counter(row["run"] for row in flagged)),
        "baseline_continuation_3090": baseline_summary,
        "xvector_4090": xvector_summary,
        "xvector_vs_baseline_per_nfe": per_nfe_change,
        "xvector_overall_duration_median_s": median([r["duration_seconds"] for r in xvector]),
        "baseline_overall_duration_median_s": median([r["duration_seconds"] for r in baseline]),
    }


def main() -> int:
    flagged = flag_rows(load_manifest())
    with FLAGS_PATH.open("w", newline="", encoding="utf-8") as flags_file:
        writer = csv.DictWriter(flags_file, fieldnames=FLAG_FIELDS)
        writer.writeheader()
        writer.writerows(flagged)
    summary = build_summary(flagged)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(flagged)} rows to {FLAGS_PATH.relative_to(ROOT)}")
    for run, count in sorted(summary["rows_per_run"].items()):
        statuses = Counter(row["status"] for row in flagged if row["run"] == run)
        print(f"  {run}: {count} rows {dict(statuses)}")
    print(f"Wrote {SUMMARY_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
