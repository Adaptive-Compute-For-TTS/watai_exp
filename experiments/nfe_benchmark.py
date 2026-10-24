"""Benchmark dots.tts generation latency across NFE settings.

``NFE`` is the number of Euler vector-field evaluations per generated latent
audio patch. The script records one row per prompt, NFE, and measured repeat.
Latency covers the synchronized ``runtime.generate`` call. It includes input
preparation, latent generation, and vocoding, but excludes model loading,
warmups, CPU transfer, and WAV writing.

Run from the repository root, for example:

    python experiments/nfe_benchmark.py --optimize --precision bfloat16

For a small CPU correctness smoke run, use one prompt and one NFE:

    python experiments/nfe_benchmark.py --precision float32 --nfe-values 4 \
        --prompt-limit 1 --repeats 1 --warmup-runs-per-case 0 \
        --output-dir /tmp/dots-tts-smoke

The default prompt manifest is ``experiments/benchmark_prompts_v1.csv``. The
shared reference audio and transcript default to the prior sweep's values.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_NFE_VALUES = (4, 6, 8, 10, 12, 16, 20, 24, 32)
DEFAULT_MODEL = "dots-studio/dots.tts-soar"
DEFAULT_MANIFEST = Path(__file__).with_name("benchmark_prompts_v1.csv")
DEFAULT_REFERENCE_AUDIO = (
    Path(__file__).resolve().with_name("my_files") / "abhi1.wav"
)
DEFAULT_REFERENCE_TEXT = (
    "Hey, Adam here. Let's create something that feels real, sounds human, "
    "and connects every time."
)
RESULT_FIELDS = (
    "prompt_id",
    "NFE",
    "repeat",
    "latency_seconds",
    "audio_duration_seconds",
    "RTF",
    "sample_rate",
    "output_path",
    "model",
    "device",
    "precision",
    "optimize",
    "seed",
    "ode_method",
    "guidance_scale",
    "speaker_scale",
    "max_generate_length",
    "max_sequence_length",
)
SNAPSHOT_FIELDS = ("prompt_id", "text", "reference_audio", "reference_text")


@dataclass(frozen=True)
class BenchmarkPrompt:
    prompt_id: str
    text: str
    reference_audio: Path | None
    reference_text: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate fixed prompts at several NFE values and log latency."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help=f"CSV or JSON prompt manifest (default: {DEFAULT_MANIFEST})",
    )
    parser.add_argument(
        "--reference-audio",
        type=Path,
        default=DEFAULT_REFERENCE_AUDIO,
        help="Shared reference audio for manifest rows without reference_audio",
    )
    parser.add_argument(
        "--reference-text",
        default=DEFAULT_REFERENCE_TEXT,
        help="Shared transcript for manifest rows without reference_text",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Local model directory or Hugging Face model ID",
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--nfe-values",
        nargs="+",
        type=int,
        default=list(DEFAULT_NFE_VALUES),
        metavar="N",
        help="NFE values to benchmark (default: 4 6 8 10 12 16 20 24 32)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--ode-method",
        choices=("euler",),
        default="euler",
        help="NFE sweep uses fixed-step Euler so each setting is one field evaluation per step",
    )
    parser.add_argument("--guidance-scale", type=float, default=1.2)
    parser.add_argument("--speaker-scale", type=float, default=1.5)
    parser.add_argument(
        "--voice-conditioning",
        choices=("x-vector", "continuation"),
        default="x-vector",
        help=(
            "Use reference audio for speaker identity only (x-vector), or also "
            "condition on its transcript as continuation context"
        ),
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Measured runs per prompt/NFE pair (default: 3)",
    )
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument(
        "--optimize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use the optimized inference path (default: disabled)",
    )
    parser.add_argument(
        "--max-generate-length",
        type=int,
        default=500,
        help=(
            "Latent patch budget per generation (default: 500, the upstream "
            "default). Archived runs used 80, which truncated long prompts"
        ),
    )
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--warmup-runs-per-case",
        type=int,
        default=1,
        help=(
            "Unmeasured warmup generations for each unit selected by "
            "--warmup-policy; set 0 to skip"
        ),
    )
    parser.add_argument(
        "--warmup-policy",
        choices=("per-prompt-nfe", "representative-prompt"),
        default="per-prompt-nfe",
        help=(
            "Warm each pending prompt/NFE pair (current behavior), or warm each "
            "pending NFE once using the first selected prompt"
        ),
    )
    parser.add_argument(
        "--prompt-limit",
        type=int,
        default=None,
        help="Use only the first N prompts, useful for a quick correctness smoke run",
    )
    parser.add_argument(
        "--prompt-offset",
        type=int,
        default=0,
        help="Skip the first N manifest prompts before applying --prompt-limit",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/nfe_benchmark_runs/run_TIMESTAMP)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace results.csv and WAV files already in the output directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an existing run, skipping prompt/NFE/repeat cases already saved",
    )
    args = parser.parse_args(argv)

    if not args.nfe_values or any(nfe <= 0 for nfe in args.nfe_values):
        parser.error("--nfe-values must contain positive integers")
    if len(set(args.nfe_values)) != len(args.nfe_values):
        parser.error("--nfe-values cannot contain duplicates")
    if args.warmup_runs_per_case < 0:
        parser.error("--warmup-runs-per-case cannot be negative")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.prompt_limit is not None and args.prompt_limit <= 0:
        parser.error("--prompt-limit must be positive")
    if args.prompt_offset < 0:
        parser.error("--prompt-offset cannot be negative")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite cannot be used together")
    if args.max_generate_length <= 0 or args.max_sequence_length <= 0:
        parser.error("generation and sequence length limits must be positive")
    return args


def load_prompts(
    manifest_path: Path,
    *,
    reference_audio: Path | None = None,
    reference_text: str | None = None,
) -> list[BenchmarkPrompt]:
    manifest_path = manifest_path.expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prompt manifest does not exist: {manifest_path}")

    if manifest_path.suffix.lower() == ".json":
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            data = data.get("prompts")
        if not isinstance(data, list):
            raise ValueError(
                "JSON prompt manifest must be a list of prompt objects or an object "
                "with a 'prompts' list"
            )
        rows = list(enumerate(data, start=1))
    else:
        with manifest_path.open(
            "r", newline="", encoding="utf-8-sig"
        ) as manifest_file:
            rows = list(enumerate(csv.DictReader(manifest_file), start=2))

    prompts: list[BenchmarkPrompt] = []
    for row_number, row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"Manifest row {row_number} must be an object")

        id_value = row.get("prompt_id") or row.get("id")
        text_value = row.get("text")
        prompt_id = "" if id_value is None else str(id_value).strip()
        text = "" if text_value is None else str(text_value).strip()
        row_audio_value = row.get("reference_audio")
        row_audio_value = (
            "" if row_audio_value is None else str(row_audio_value).strip()
        )
        row_reference_text = row.get("reference_text")
        row_reference_text = (
            "" if row_reference_text is None else str(row_reference_text).strip()
        )
        resolved_reference_text = row_reference_text or reference_text or None

        if not prompt_id or not text:
            raise ValueError(
                f"Manifest row {row_number} must have non-empty id/prompt_id and text"
            )

        reference_audio_path = None
        audio_value = row_audio_value or reference_audio
        if audio_value:
            reference_audio_path = Path(audio_value).expanduser()
            if not reference_audio_path.is_absolute():
                base_dir = manifest_path.parent if row_audio_value else Path.cwd()
                reference_audio_path = base_dir / reference_audio_path
            reference_audio_path = reference_audio_path.resolve()
            if not reference_audio_path.is_file():
                raise FileNotFoundError(
                    f"Reference audio on manifest row {row_number} does not exist: "
                    f"{reference_audio_path}"
                )
        elif resolved_reference_text:
            raise ValueError(
                f"Manifest row {row_number} has reference_text but no reference_audio"
            )

        prompts.append(
            BenchmarkPrompt(
                prompt_id=prompt_id,
                text=text,
                reference_audio=reference_audio_path,
                reference_text=resolved_reference_text,
            )
        )

    if not prompts:
        raise ValueError(f"Prompt manifest contains no prompts: {manifest_path}")
    ids = [prompt.prompt_id for prompt in prompts]
    if len(set(ids)) != len(ids):
        raise ValueError("prompt_id values in the manifest must be unique")
    return prompts


def safe_prompt_id(prompt_id: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9._-]+", "_", prompt_id).strip("._")
    if not safe_id:
        raise ValueError(f"prompt_id cannot be converted to a filename: {prompt_id!r}")
    return safe_id


def output_wav_path(
    output_dir: Path,
    prompt_id: str,
    nfe: int,
    repeat: int,
    repeats: int,
) -> Path:
    repeat_suffix = "" if repeats == 1 else f"_rep{repeat}"
    filename = f"output_{safe_prompt_id(prompt_id)}_nfe{nfe}{repeat_suffix}.wav"
    return output_dir / filename


def synchronize_if_cuda(torch: Any, runtime: Any) -> None:
    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)


def prompt_snapshot_rows(prompts: list[BenchmarkPrompt]) -> list[dict[str, str]]:
    return [
        {
            "prompt_id": prompt.prompt_id,
            "text": prompt.text,
            "reference_audio": (
                "" if prompt.reference_audio is None else str(prompt.reference_audio)
            ),
            "reference_text": prompt.reference_text or "",
        }
        for prompt in prompts
    ]


def write_prompt_snapshot(
    path: Path, prompts: list[BenchmarkPrompt], *, overwrite: bool = False
) -> None:
    mode = "w" if overwrite else "x"
    with path.open(mode, newline="", encoding="utf-8") as snapshot_file:
        writer = csv.DictWriter(snapshot_file, fieldnames=SNAPSHOT_FIELDS)
        writer.writeheader()
        writer.writerows(prompt_snapshot_rows(prompts))


def validate_prompt_snapshot(path: Path, prompts: list[BenchmarkPrompt]) -> None:
    if not path.is_file():
        raise ValueError(
            f"Cannot resume without the original prompt snapshot: {path}"
        )
    with path.open("r", newline="", encoding="utf-8") as snapshot_file:
        reader = csv.DictReader(snapshot_file)
        if tuple(reader.fieldnames or ()) != SNAPSHOT_FIELDS:
            raise ValueError(f"Prompt snapshot has unexpected columns: {path}")
        actual_rows = list(reader)
    expected_rows = prompt_snapshot_rows(prompts)
    if actual_rows != expected_rows:
        raise ValueError(
            "The selected prompts or reference audio/text differ from the saved run. "
            "Use the original settings/output directory or choose a new output directory."
        )


def resume_config(args: argparse.Namespace) -> dict[str, str]:
    return {
        "model": str(args.model),
        "precision": str(args.precision),
        "optimize": str(args.optimize),
        "seed": str(args.seed),
        "ode_method": str(args.ode_method),
        "guidance_scale": str(args.guidance_scale),
        "speaker_scale": str(args.speaker_scale),
        "max_generate_length": str(args.max_generate_length),
        "max_sequence_length": str(args.max_sequence_length),
    }


def run_metadata(
    args: argparse.Namespace,
    prompts: list[BenchmarkPrompt],
) -> dict[str, Any]:
    representative_prompt_id = None
    if args.warmup_policy == "representative-prompt" and args.warmup_runs_per_case:
        representative_prompt_id = prompts[0].prompt_id
    wav_repeat_suffix = "" if args.repeats == 1 else "_rep{repeat}"
    return {
        "metadata_version": 1,
        "manifest_path": str(args.manifest.expanduser().resolve()),
        "prompt_count": len(prompts),
        "prompt_ids": [prompt.prompt_id for prompt in prompts],
        "prompt_snapshot": "prompts.csv",
        "nfe_values": list(args.nfe_values),
        "measured_repeats_per_prompt_nfe": args.repeats,
        "model": args.model,
        "voice_conditioning": args.voice_conditioning,
        "revision": args.revision,
        "cache_dir": None if args.cache_dir is None else str(args.cache_dir),
        "precision": args.precision,
        "optimize": args.optimize,
        "seed": args.seed,
        "ode_method": args.ode_method,
        "guidance_scale": args.guidance_scale,
        "speaker_scale": args.speaker_scale,
        "max_generate_length": args.max_generate_length,
        "max_sequence_length": args.max_sequence_length,
        "warmup": {
            "policy": args.warmup_policy,
            "runs_per_unit": args.warmup_runs_per_case,
            "unit": (
                "NFE using representative prompt"
                if args.warmup_policy == "representative-prompt"
                else "prompt/NFE pair"
            ),
            "representative_prompt_id": representative_prompt_id,
            "measured": False,
            "save_wav": False,
        },
        "output": {
            "results_file": "results.csv",
            "audio_format": "WAV",
            "audio_filename_pattern": (
                f"output_{{prompt_id}}_nfe{{nfe}}{wav_repeat_suffix}.wav"
            ),
        },
    }


def write_run_metadata(
    path: Path,
    metadata: dict[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    payload = {
        **metadata,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as metadata_file:
        json.dump(payload, metadata_file, indent=2)
        metadata_file.write("\n")


def validate_run_metadata(path: Path, expected: dict[str, Any]) -> None:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read run metadata: {path}") from exc
    actual.pop("created_at_utc", None)
    if actual != expected:
        raise ValueError(
            "Run metadata does not match the requested benchmark settings. "
            "Use the original settings/output directory or choose a new output directory."
        )


def load_resume_state(
    results_path: Path,
    *,
    prompts: list[BenchmarkPrompt],
    nfe_values: list[int],
    repeats: int,
    output_dir: Path,
    args: argparse.Namespace,
) -> tuple[set[tuple[str, int, int]], list[dict[str, str]]]:
    """Validate saved rows and return cases with both a CSV row and WAV file."""
    if not results_path.exists():
        return set(), []

    raw_lines = results_path.read_bytes().splitlines(keepends=True)
    if not raw_lines or not raw_lines[0].endswith(b"\n"):
        raise ValueError(f"Existing results file has no complete CSV header: {results_path}")
    complete_lines = raw_lines
    dropped_partial_line = False
    if not raw_lines[-1].endswith(b"\n"):
        complete_lines = raw_lines[:-1]
        dropped_partial_line = True

    decoded = b"".join(complete_lines).decode("utf-8")
    reader = csv.DictReader(decoded.splitlines())
    if tuple(reader.fieldnames or ()) != RESULT_FIELDS:
        raise ValueError(f"Results file has unexpected columns: {results_path}")

    expected_cases = {
        (prompt.prompt_id, nfe, repeat)
        for prompt in prompts
        for nfe in nfe_values
        for repeat in range(1, repeats + 1)
    }
    config = resume_config(args)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, int, int]] = set()
    completed: set[tuple[str, int, int]] = set()
    rows_need_rewrite = dropped_partial_line

    for row in reader:
        if None in row or any(row.get(field) is None for field in RESULT_FIELDS):
            raise ValueError(
                "Results file contains a malformed complete CSV row; "
                "repair it before resuming."
            )
        try:
            case = (row["prompt_id"], int(row["NFE"]), int(row["repeat"]))
        except ValueError as exc:
            raise ValueError("Results file contains an invalid NFE/repeat value") from exc
        if case not in expected_cases:
            raise ValueError(
                f"Saved result {case} does not match this prompt/NFE/repeat selection"
            )
        if case in seen:
            raise ValueError(f"Results file contains a duplicate case: {case}")
        seen.add(case)
        for field, expected_value in config.items():
            if row[field] != expected_value:
                raise ValueError(
                    f"Cannot resume because saved {field}={row[field]!r}, "
                    f"but current setting is {expected_value!r}"
                )

        expected_wav = output_wav_path(output_dir, case[0], case[1], case[2], repeats)
        saved_wav = expected_wav
        if row["output_path"] != str(expected_wav):
            # Preserve a completed first pass from a run originally configured
            # with --repeats 1, whose WAV filename had no repeat suffix.
            legacy_first_pass = output_wav_path(output_dir, case[0], case[1], 1, 1)
            if (
                repeats > 1
                and case[2] == 1
                and row["output_path"] == str(legacy_first_pass)
            ):
                saved_wav = legacy_first_pass
            else:
                raise ValueError(
                    f"Saved output path does not match this run: {row['output_path']}"
                )
        if not saved_wav.is_file():
            # A saved row without its audio is incomplete and should rerun.
            rows_need_rewrite = True
            continue
        rows.append(row)
        completed.add(case)

    if rows_need_rewrite:
        temporary_path = results_path.with_name(results_path.name + ".resume.tmp")
        with temporary_path.open("w", newline="", encoding="utf-8") as results_file:
            writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            results_file.flush()
        temporary_path.replace(results_path)
    return completed, rows


def generate_once(
    *,
    runtime: Any,
    torch: Any,
    seed_everything: Any,
    prompt: BenchmarkPrompt,
    nfe: int,
    seed: int,
    ode_method: str,
    guidance_scale: float,
    speaker_scale: float,
    voice_conditioning: str,
) -> tuple[dict[str, Any], float]:
    seed_everything(seed)
    synchronize_if_cuda(torch, runtime)
    started = time.perf_counter()
    result = runtime.generate(
        text=prompt.text,
        prompt_audio_path=(
            None if prompt.reference_audio is None else str(prompt.reference_audio)
        ),
        prompt_text=(
            prompt.reference_text if voice_conditioning == "continuation" else None
        ),
        num_steps=nfe,
        ode_method=ode_method,
        guidance_scale=guidance_scale,
        speaker_scale=speaker_scale,
    )
    synchronize_if_cuda(torch, runtime)
    return result, time.perf_counter() - started


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompts = load_prompts(
        args.manifest,
        reference_audio=args.reference_audio,
        reference_text=args.reference_text,
    )
    manifest_prompt_count = len(prompts)
    prompts = prompts[args.prompt_offset :]
    if args.prompt_limit is not None:
        prompts = prompts[: args.prompt_limit]
    if not prompts:
        raise ValueError("--prompt-offset and --prompt-limit select no prompts")
    safe_ids = [safe_prompt_id(prompt.prompt_id) for prompt in prompts]
    if len(set(safe_ids)) != len(safe_ids):
        raise ValueError("prompt_id values must produce unique output filenames")
    print(
        f"Prompt manifest: {args.manifest.expanduser().resolve()} "
        f"({manifest_prompt_count} rows); selected {len(prompts)} prompts: "
        f"{prompts[0].prompt_id} through {prompts[-1].prompt_id}"
    )

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or (
        Path(__file__).with_name("nfe_benchmark_runs") / f"run_{run_stamp}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    manifest_snapshot_path = output_dir / "prompts.csv"
    run_metadata_path = output_dir / "run_metadata.json"
    expected_metadata = run_metadata(args, prompts)
    expected_wavs = [
        output_wav_path(
            output_dir,
            prompt.prompt_id,
            nfe,
            repeat,
            args.repeats,
        )
        for prompt in prompts
        for nfe in args.nfe_values
        for repeat in range(1, args.repeats + 1)
    ]
    completed_cases: set[tuple[str, int, int]] = set()
    if args.resume:
        if run_metadata_path.exists():
            validate_run_metadata(run_metadata_path, expected_metadata)
        elif not results_path.exists():
            # Legacy pilot runs have no metadata. Preserve their existing output
            # directories when resuming them, while fresh runs get full metadata.
            write_run_metadata(run_metadata_path, expected_metadata)
        if manifest_snapshot_path.exists():
            validate_prompt_snapshot(manifest_snapshot_path, prompts)
        elif results_path.exists():
            validate_prompt_snapshot(manifest_snapshot_path, prompts)
        else:
            write_prompt_snapshot(manifest_snapshot_path, prompts)
        completed_cases, _ = load_resume_state(
            results_path,
            prompts=prompts,
            nfe_values=args.nfe_values,
            repeats=args.repeats,
            output_dir=output_dir,
            args=args,
        )
        total_cases = len(prompts) * len(args.nfe_values) * args.repeats
        print(
            f"Resume state: {len(completed_cases)}/{total_cases} cases complete; "
            f"{total_cases - len(completed_cases)} remaining"
        )
        if len(completed_cases) == total_cases:
            print(f"All cases already complete in {results_path}")
            return 0
    else:
        existing_outputs = [
            path
            for path in [
                results_path,
                manifest_snapshot_path,
                run_metadata_path,
                *expected_wavs,
            ]
            if path.exists()
        ]
        if existing_outputs and not args.overwrite:
            paths = "\n".join(str(path) for path in existing_outputs)
            raise FileExistsError(
                "Output files already exist. Choose another --output-dir, "
                "pass --resume to continue, or pass --overwrite to replace them:\n"
                f"{paths}"
            )
        write_prompt_snapshot(manifest_snapshot_path, prompts, overwrite=args.overwrite)
        write_run_metadata(
            run_metadata_path,
            expected_metadata,
            overwrite=args.overwrite,
        )

    pending_pairs = {
        (prompt.prompt_id, nfe)
        for prompt in prompts
        for nfe in args.nfe_values
        if any(
            (prompt.prompt_id, nfe, repeat) not in completed_cases
            for repeat in range(1, args.repeats + 1)
        )
    }

    import soundfile as sf
    import torch

    from dots_tts.runtime import DotsTtsRuntime
    from dots_tts.utils.util import seed_everything

    runtime = DotsTtsRuntime.from_pretrained(
        args.model,
        revision=args.revision,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir),
        precision=args.precision,
        optimize=args.optimize,
        max_generate_length=args.max_generate_length,
        max_sequence_length=args.max_sequence_length,
    )
    samples_per_patch = int(runtime.model.config.patch_size * runtime.model.hop_size)
    print(
        f"Loaded {args.model} on {runtime.device} with "
        f"{args.voice_conditioning} voice conditioning; writing results under {output_dir}"
    )

    if args.warmup_runs_per_case:
        if args.warmup_policy == "representative-prompt":
            pending_nfes = [
                nfe
                for nfe in args.nfe_values
                if any(pair_nfe == nfe for _, pair_nfe in pending_pairs)
            ]
            warmup_prompt = prompts[0]
            print(
                f"Running {args.warmup_runs_per_case} unmeasured warmup(s) "
                f"per pending NFE using prompt {warmup_prompt.prompt_id}; "
                "warmups are not saved"
            )
            for nfe in pending_nfes:
                for _ in range(args.warmup_runs_per_case):
                    generate_once(
                        runtime=runtime,
                        torch=torch,
                        seed_everything=seed_everything,
                        prompt=warmup_prompt,
                        nfe=nfe,
                        seed=args.seed,
                        ode_method=args.ode_method,
                        guidance_scale=args.guidance_scale,
                        speaker_scale=args.speaker_scale,
                        voice_conditioning=args.voice_conditioning,
                    )
        else:
            print(
                f"Running {args.warmup_runs_per_case} unmeasured warmup(s) "
                "for each pending prompt/NFE pair; warmups are not saved"
            )
            for prompt in prompts:
                for nfe in args.nfe_values:
                    if (prompt.prompt_id, nfe) not in pending_pairs:
                        continue
                    for _ in range(args.warmup_runs_per_case):
                        generate_once(
                            runtime=runtime,
                            torch=torch,
                            seed_everything=seed_everything,
                            prompt=prompt,
                            nfe=nfe,
                            seed=args.seed,
                            ode_method=args.ode_method,
                            guidance_scale=args.guidance_scale,
                            speaker_scale=args.speaker_scale,
                            voice_conditioning=args.voice_conditioning,
                        )

    results_existed = results_path.exists()
    if args.resume:
        open_mode = "a" if results_existed else "x"
    else:
        open_mode = "w" if args.overwrite else "x"
    with results_path.open(open_mode, newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
        if not args.resume or not results_existed:
            writer.writeheader()
            results_file.flush()

        for prompt in prompts:
            for repeat in range(1, args.repeats + 1):
                # Reverse the sweep order on alternating repeats to reduce
                # systematic timing bias from always measuring low NFE first.
                nfe_order = (
                    args.nfe_values
                    if repeat % 2 == 1
                    else tuple(reversed(args.nfe_values))
                )
                for nfe in nfe_order:
                    case = (prompt.prompt_id, nfe, repeat)
                    if case in completed_cases:
                        continue
                    output_path = output_wav_path(
                        output_dir,
                        prompt.prompt_id,
                        nfe,
                        repeat,
                        args.repeats,
                    )
                    if output_path.exists() and not (args.overwrite or args.resume):
                        raise FileExistsError(f"Output WAV already exists: {output_path}")

                    result, latency = generate_once(
                        runtime=runtime,
                        torch=torch,
                        seed_everything=seed_everything,
                        prompt=prompt,
                        nfe=nfe,
                        seed=args.seed,
                        ode_method=args.ode_method,
                        guidance_scale=args.guidance_scale,
                        speaker_scale=args.speaker_scale,
                        voice_conditioning=args.voice_conditioning,
                    )
                    audio_duration = (
                        int(result["audio"].shape[-1]) / result["sample_rate"]
                    )
                    audio = result["audio"].detach().float().cpu().squeeze().numpy()
                    rtf = latency / audio_duration if audio_duration > 0 else float("inf")
                    generated_patches = int(result["audio"].shape[-1]) // samples_per_patch
                    if (
                        args.voice_conditioning == "x-vector"
                        and generated_patches >= args.max_generate_length
                    ):
                        print(
                            f"WARNING: prompt={prompt.prompt_id} NFE={nfe} reached "
                            f"max_generate_length={args.max_generate_length}; "
                            "the output is likely truncated"
                        )
                    sf.write(output_path, audio, result["sample_rate"])

                    writer.writerow(
                        {
                            "prompt_id": prompt.prompt_id,
                            "NFE": nfe,
                            "repeat": repeat,
                            "latency_seconds": f"{latency:.6f}",
                            "audio_duration_seconds": f"{audio_duration:.6f}",
                            "RTF": f"{rtf:.6f}",
                            "sample_rate": result["sample_rate"],
                            "output_path": str(output_path),
                            "model": args.model,
                            "device": str(runtime.device),
                            "precision": args.precision,
                            "optimize": args.optimize,
                            "seed": args.seed,
                            "ode_method": args.ode_method,
                            "guidance_scale": args.guidance_scale,
                            "speaker_scale": args.speaker_scale,
                            "max_generate_length": args.max_generate_length,
                            "max_sequence_length": args.max_sequence_length,
                        }
                    )
                    results_file.flush()
                    completed_cases.add(case)
                    print(
                        f"prompt={prompt.prompt_id} NFE={nfe} repeat={repeat} "
                        f"latency={latency:.3f}s audio={audio_duration:.3f}s "
                        f"RTF={rtf:.3f} -> {output_path.name}"
                    )

    print(f"CSV results saved to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
