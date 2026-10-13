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

The default prompt manifest is ``experiments/nfe_prompts.csv``. Its reference
audio paths are resolved relative to the manifest file.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_NFE_VALUES = (4, 8, 10, 16, 24, 32)
DEFAULT_MODEL = "dots-studio/dots.tts-soar"
DEFAULT_MANIFEST = Path(__file__).with_name("nfe_prompts.csv")
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
        default=None,
        help="Shared reference audio for manifest rows without reference_audio",
    )
    parser.add_argument(
        "--reference-text",
        default=None,
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
        help="NFE values to benchmark (default: 4 8 10 16 24 32)",
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
    parser.add_argument("--max-generate-length", type=int, default=80)
    parser.add_argument("--max-sequence-length", type=int, default=2048)
    parser.add_argument(
        "--warmup-runs-per-case",
        type=int,
        default=1,
        help="Unmeasured warmup generations for every prompt/NFE pair; set 0 to skip",
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
) -> tuple[dict[str, Any], float]:
    seed_everything(seed)
    synchronize_if_cuda(torch, runtime)
    started = time.perf_counter()
    result = runtime.generate(
        text=prompt.text,
        prompt_audio_path=(
            None if prompt.reference_audio is None else str(prompt.reference_audio)
        ),
        prompt_text=prompt.reference_text,
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
    prompts = prompts[args.prompt_offset :]
    if args.prompt_limit is not None:
        prompts = prompts[: args.prompt_limit]
    if not prompts:
        raise ValueError("--prompt-offset and --prompt-limit select no prompts")
    safe_ids = [safe_prompt_id(prompt.prompt_id) for prompt in prompts]
    if len(set(safe_ids)) != len(safe_ids):
        raise ValueError("prompt_id values must produce unique output filenames")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = args.output_dir or (
        Path(__file__).with_name("nfe_benchmark_runs") / f"run_{run_stamp}"
    )
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    manifest_snapshot_path = output_dir / "prompts.csv"
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
    existing_outputs = [
        path
        for path in [results_path, manifest_snapshot_path, *expected_wavs]
        if path.exists()
    ]
    if existing_outputs and not args.overwrite:
        paths = "\n".join(str(path) for path in existing_outputs)
        raise FileExistsError(
            "Output files already exist. Choose another --output-dir or pass "
            f"--overwrite:\n{paths}"
        )

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
    print(f"Loaded {args.model} on {runtime.device}; writing results under {output_dir}")

    snapshot_mode = "w" if args.overwrite else "x"
    with manifest_snapshot_path.open(
        snapshot_mode, newline="", encoding="utf-8"
    ) as snapshot_file:
        snapshot_writer = csv.DictWriter(
            snapshot_file,
            fieldnames=("prompt_id", "text", "reference_audio", "reference_text"),
        )
        snapshot_writer.writeheader()
        for prompt in prompts:
            snapshot_writer.writerow(
                {
                    "prompt_id": prompt.prompt_id,
                    "text": prompt.text,
                    "reference_audio": (
                        "" if prompt.reference_audio is None else prompt.reference_audio
                    ),
                    "reference_text": prompt.reference_text or "",
                }
            )

    if args.warmup_runs_per_case:
        print(
            f"Running {args.warmup_runs_per_case} unmeasured warmup(s) "
            "for each prompt/NFE pair"
        )
        for prompt in prompts:
            for nfe in args.nfe_values:
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
                    )

    open_mode = "w" if args.overwrite else "x"
    with results_path.open(open_mode, newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=RESULT_FIELDS)
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
                    output_path = output_wav_path(
                        output_dir,
                        prompt.prompt_id,
                        nfe,
                        repeat,
                        args.repeats,
                    )
                    if output_path.exists() and not args.overwrite:
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
                    )
                    audio_duration = (
                        int(result["audio"].shape[-1]) / result["sample_rate"]
                    )
                    audio = result["audio"].detach().float().cpu().squeeze().numpy()
                    rtf = latency / audio_duration if audio_duration > 0 else float("inf")
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
                    print(
                        f"prompt={prompt.prompt_id} NFE={nfe} repeat={repeat} "
                        f"latency={latency:.3f}s audio={audio_duration:.3f}s "
                        f"RTF={rtf:.3f} -> {output_path.name}"
                    )

    print(f"CSV results saved to {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
