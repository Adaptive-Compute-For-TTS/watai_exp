"""Plot and compare the archived 54-prompt NFE and x-vector sweeps.

Run from the repository root:
    python experiments/plot_nfe_sweep_comparison.py

The script reads the four archived part directories, validates prompt and
configuration alignment, and writes SVG figures plus an analysis summary.
"""

from __future__ import annotations

import csv
import hashlib
import html
import json
import math
import os
import statistics
import wave
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "experiments" / "nfe_benchmark_runs"
BASELINE_PARTS = [
    RUN_ROOT / "nfe_54_prompts_v1_full_part_0",
    RUN_ROOT / "nfe_54_prompts_v1_full_part_1",
]
XVECTOR_PARTS = [
    RUN_ROOT / "nfe_54_prompts_v1_full_xvector_part_0",
    RUN_ROOT / "nfe_54_prompts_v1_full_xvector_part_1",
]
PILOT_GRAPHS = ROOT / "graphs" / "01_pilot_rtx4090_single_prompt"
XVECTOR_GRAPHS = ROOT / "graphs" / "02_full_rtx4090_xvector_54_prompts"
COMPARISON_GRAPHS = ROOT / "graphs" / "03_full_sweep_rtx3090_vs_rtx4090"
ANALYSIS_DIR = RUN_ROOT / "analysis" / "full_sweep_xvector_comparison"
CLEAN_SAMPLES_SUMMARY = ROOT / "experiments" / "clean_samples" / "run_summary.json"

NFES = [4, 6, 8, 10, 12, 16, 20, 24, 32]
# One latent patch is 0.16 s of 48 kHz audio; max_generate_length counts patches.
PATCH_SECONDS = 0.16
CAP_TOLERANCE_SECONDS = 0.01
BASELINE_LABEL = "Original run, RTX 3090 (reported)"
XVECTOR_LABEL = "X-vector run, RTX 4090 (reported)"
BASELINE_COLOR = "#2563eb"
XVECTOR_COLOR = "#ea580c"
DOT_COLOR = "#94a3b8"
GRID_COLOR = "#e2e8f0"
TEXT_COLOR = "#1e293b"
MUTED_COLOR = "#64748b"
THRESHOLD_COLOR = "#dc2626"

METRICS = {
    "latency": {
        "field": "latency_seconds",
        "title": "NFE vs generation latency",
        "ylabel": "Latency (seconds)",
        "subtitle": "Wall-clock time per generated WAV across the prompt set.",
        "xvector_max": 60.0,
        "xvector_tick": 10.0,
        "comparison_max": 120.0,
        "comparison_tick": 20.0,
    },
    "rtf": {
        "field": "RTF",
        "title": "NFE vs real-time factor",
        "ylabel": "Real-time factor (RTF)",
        "subtitle": "RTF = latency / WAV duration. Values below 1 are faster than playback.",
        "xvector_max": 6.0,
        "xvector_tick": 1.0,
        "comparison_max": 14.0,
        "comparison_tick": 2.0,
        "threshold": 1.0,
    },
    "duration": {
        "field": "audio_duration_seconds",
        "title": "NFE vs generated audio duration",
        "ylabel": "WAV duration (seconds)",
        "subtitle": "Generated clip length across prompts. Duration alone does not verify spoken content.",
        "xvector_max": 14.0,
        "xvector_tick": 2.0,
        "comparison_max": 14.0,
        "comparison_tick": 2.0,
    },
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def read_partition(folder: Path) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    metadata_path = folder / "run_metadata.json"
    prompt_path = folder / "prompts.csv"
    results_path = folder / "results.csv"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    prompts = {row["prompt_id"]: row["text"] for row in read_csv(prompt_path)}
    rows: list[dict[str, Any]] = []
    wav_hashes_by_nfe: dict[int, set[str]] = defaultdict(set)
    for row in read_csv(results_path):
        row["NFE"] = int(row["NFE"])
        row["repeat"] = int(row["repeat"])
        for field in ("latency_seconds", "audio_duration_seconds", "RTF"):
            row[field] = float(row[field])
        wav_path = folder / Path(row["output_path"]).name
        if not wav_path.is_file():
            raise ValueError(f"Missing result WAV: {wav_path}")
        with wave.open(str(wav_path), "rb") as audio:
            frames = audio.readframes(audio.getnframes())
            wav_duration = len(frames) / (
                audio.getframerate() * audio.getsampwidth() * audio.getnchannels()
            )
        if abs(wav_duration - row["audio_duration_seconds"]) > 0.001:
            raise ValueError(f"WAV duration disagrees with results.csv: {wav_path}")
        wav_hashes_by_nfe[row["NFE"]].add(hashlib.sha256(frames).hexdigest())
        rows.append(row)
    for nfe in NFES:
        nfe_rows = [row for row in rows if row["NFE"] == nfe]
        if len(nfe_rows) == 27 and len(wav_hashes_by_nfe[nfe]) != 27:
            raise ValueError(
                f"{folder.name}: NFE {nfe} has duplicate waveform samples across prompt WAVs"
            )
    return metadata, prompts, rows


def load_pair(parts: list[Path], label: str) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    metadata_parts = []
    prompt_parts = []
    rows: list[dict[str, Any]] = []
    for folder in parts:
        metadata, prompts, part_rows = read_partition(folder)
        metadata_parts.append(metadata)
        prompt_parts.append(prompts)
        rows.extend(part_rows)

    if metadata_parts[0]["nfe_values"] != NFES or metadata_parts[1]["nfe_values"] != NFES:
        raise ValueError(f"{label}: unexpected NFE settings")
    if len(prompt_parts[0]) != 27 or len(prompt_parts[1]) != 27:
        raise ValueError(f"{label}: expected 27 prompts in each part")
    if set(prompt_parts[0]) & set(prompt_parts[1]):
        raise ValueError(f"{label}: the two parts contain overlapping prompt IDs")

    prompts = prompt_parts[0] | prompt_parts[1]
    if len(prompts) != 54:
        raise ValueError(f"{label}: expected 54 unique prompt snapshots, found {len(prompts)}")
    metadata_ids = set(metadata_parts[0]["prompt_ids"]) | set(metadata_parts[1]["prompt_ids"])
    if metadata_ids != set(prompts):
        raise ValueError(f"{label}: metadata prompt IDs do not match prompts.csv")

    seen = Counter((row["prompt_id"], row["NFE"]) for row in rows)
    expected = {(prompt_id, nfe) for prompt_id in prompts for nfe in NFES}
    if set(seen) != expected or set(seen.values()) != {1}:
        raise ValueError(f"{label}: results must contain one row per prompt/NFE pair")
    if len(rows) != 54 * len(NFES):
        raise ValueError(f"{label}: expected 486 measured rows, found {len(rows)}")
    if any(row["repeat"] != 1 for row in rows):
        raise ValueError(f"{label}: expected one measured generation per prompt/NFE")

    common_fields = (
        "model", "precision", "optimize", "seed", "ode_method", "guidance_scale",
        "speaker_scale", "max_generate_length", "max_sequence_length",
        "nfe_values", "measured_repeats_per_prompt_nfe",
    )
    for field in common_fields:
        if metadata_parts[0].get(field) != metadata_parts[1].get(field):
            raise ValueError(f"{label}: run parts disagree on metadata field {field!r}")
    warmup_fields = ("policy", "runs_per_unit", "unit", "measured", "save_wav")
    for field in warmup_fields:
        left = metadata_parts[0].get("warmup", {}).get(field)
        right = metadata_parts[1].get("warmup", {}).get(field)
        if left != right:
            raise ValueError(f"{label}: run parts disagree on warmup field {field!r}")

    rows.sort(key=lambda row: (row["prompt_id"], row["NFE"]))
    return metadata_parts[0], prompts, rows


def validate_comparable_runs(
    baseline_meta: dict[str, Any],
    baseline_prompts: dict[str, str],
    baseline_rows: list[dict[str, Any]],
    xvector_meta: dict[str, Any],
    xvector_prompts: dict[str, str],
    xvector_rows: list[dict[str, Any]],
) -> None:
    if baseline_prompts != xvector_prompts:
        raise ValueError("The original and x-vector prompt snapshots differ")
    if baseline_meta["nfe_values"] != xvector_meta["nfe_values"]:
        raise ValueError("The runs use different NFE values")

    metadata_fields = (
        "model", "precision", "optimize", "seed", "ode_method", "guidance_scale",
        "speaker_scale", "max_generate_length", "max_sequence_length",
        "measured_repeats_per_prompt_nfe",
    )
    for field in metadata_fields:
        if baseline_meta.get(field) != xvector_meta.get(field):
            raise ValueError(f"Run metadata differs on {field!r}")
    if xvector_meta.get("voice_conditioning") != "x-vector":
        raise ValueError("The new run metadata does not identify x-vector conditioning")

    row_fields = (
        "model", "device", "precision", "optimize", "seed", "ode_method",
        "guidance_scale", "speaker_scale", "max_generate_length", "max_sequence_length",
    )
    for field in row_fields:
        old_values = {row[field] for row in baseline_rows}
        new_values = {row[field] for row in xvector_rows}
        if len(old_values) != 1 or old_values != new_values:
            raise ValueError(f"Results differ or vary unexpectedly on {field!r}")


def summarize(rows: list[dict[str, Any]]) -> dict[int, dict[str, float | int]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["NFE"]].append(row)
    summaries: dict[int, dict[str, float | int]] = {}
    for nfe, group in sorted(groups.items()):
        summaries[nfe] = {"count": len(group)}
        for metric, config in METRICS.items():
            values = [float(row[config["field"]]) for row in group]
            summaries[nfe][f"{metric}_q1"] = percentile(values, 0.25)
            summaries[nfe][f"{metric}_median"] = statistics.median(values)
            summaries[nfe][f"{metric}_q3"] = percentile(values, 0.75)
            summaries[nfe][f"{metric}_p90"] = percentile(values, 0.90)
    return summaries


def comparison_summary(
    baseline_rows: list[dict[str, Any]],
    xvector_rows: list[dict[str, Any]],
    baseline_stats: dict[int, dict[str, float | int]],
    xvector_stats: dict[int, dict[str, float | int]],
) -> list[dict[str, float | int]]:
    baseline_by_key = {(row["prompt_id"], row["NFE"]): row for row in baseline_rows}
    xvector_by_key = {(row["prompt_id"], row["NFE"]): row for row in xvector_rows}
    output: list[dict[str, float | int]] = []
    for nfe in NFES:
        old = baseline_stats[nfe]
        new = xvector_stats[nfe]
        paired_old = [baseline_by_key[key] for key in baseline_by_key if key[1] == nfe]
        paired_new = [xvector_by_key[key] for key in baseline_by_key if key[1] == nfe]
        row: dict[str, float | int] = {"NFE": nfe, "prompts": len(paired_old)}
        for metric in METRICS:
            old_median = float(old[f"{metric}_median"])
            new_median = float(new[f"{metric}_median"])
            row[f"baseline_{metric}_median"] = old_median
            row[f"xvector_{metric}_median"] = new_median
            row[f"{metric}_median_change_pct"] = (new_median / old_median - 1) * 100
            paired_ratios = [
                float(new_row[METRICS[metric]["field"]])
                / float(old_row[METRICS[metric]["field"]])
                for old_row, new_row in zip(paired_old, paired_new)
            ]
            row[f"paired_{metric}_ratio_median"] = statistics.median(paired_ratios)
        row["baseline_duration_median"] = float(old["duration_median"])
        row["xvector_duration_median"] = float(new["duration_median"])
        row["duration_median_change_s"] = float(new["duration_median"]) - float(old["duration_median"])
        output.append(row)
    return output


def svg_text(
    x: float,
    y: float,
    value: str,
    *,
    size: int = 14,
    anchor: str = "start",
    weight: str = "normal",
    fill: str = TEXT_COLOR,
    extra: str = "",
) -> str:
    return (
        f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" '
        f'text-anchor="{anchor}" font-weight="{weight}" fill="{fill}" '
        f'{extra}>{html.escape(value)}</text>'
    )


def svg_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    stroke: str = GRID_COLOR,
    width: float = 1,
    dash: str = "",
) -> str:
    dash_attr = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
        f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"{dash_attr}/>'
    )


def svg_document(title: str, description: str, width: int, height: int, body: str) -> str:
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>{html.escape(title)}</title>
<desc>{html.escape(description)}</desc>
<rect width="100%" height="100%" fill="#f8fafc"/>
<style>text{{font-family:Arial,Helvetica,sans-serif}}</style>
{body}
</svg>
'''


def fmt_value(metric: str, value: float) -> str:
    return f"{value:.2f}" if metric == "rtf" else f"{value:.1f}s"


def chart_panel(
    series: list[dict[str, Any]],
    summaries: dict[str, dict[int, dict[str, float | int]]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    metric: str,
    title: str,
    subtitle: str,
    maximum: float,
    tick: float,
    comparison: bool,
) -> str:
    left = x + 84
    right = x + width - 30
    top = y + 70
    bottom = y + height - 78
    plot_width = right - left
    plot_height = bottom - top
    x_at = lambda nfe: left + (nfe - NFES[0]) / (NFES[-1] - NFES[0]) * plot_width
    y_at = lambda value: bottom - value / maximum * plot_height
    all_prompt_ids = sorted({row["prompt_id"] for item in series for row in item["rows"]})
    prompt_rank = {prompt_id: index for index, prompt_id in enumerate(all_prompt_ids)}
    median_field = f"{metric}_median"
    q1_field = f"{metric}_q1"
    q3_field = f"{metric}_q3"
    field = METRICS[metric]["field"]
    pieces = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" '
        'rx="12" fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>',
        svg_text(x + 24, y + 30, title, size=21, weight="bold"),
        svg_text(x + 24, y + 52, subtitle, size=13, fill=MUTED_COLOR),
    ]
    for tick_index in range(math.floor(maximum / tick) + 1):
        value = tick_index * tick
        cy = y_at(value)
        pieces.extend([
            svg_line(left, cy, right, cy),
            svg_text(left - 12, cy + 5, f"{value:g}", size=12, anchor="end", fill=MUTED_COLOR),
        ])
    pieces.extend([
        svg_line(left, top, left, bottom, stroke="#475569", width=1.5),
        svg_line(left, bottom, right, bottom, stroke="#475569", width=1.5),
        svg_text(x + 22, (top + bottom) / 2, METRICS[metric]["ylabel"], size=13,
                 anchor="middle", extra=f'transform="rotate(-90 {x + 22:.2f} {(top + bottom) / 2:.2f})"'),
        svg_text((left + right) / 2, y + height - 18, "NFE (Euler steps)", size=13, anchor="middle"),
    ])

    for nfe in NFES:
        cx = x_at(nfe)
        pieces.append(svg_line(cx, bottom, cx, bottom + 5, stroke="#475569"))
        pieces.append(svg_text(cx, bottom + 23, str(nfe), size=12, anchor="middle", fill=MUTED_COLOR))

    threshold = METRICS[metric].get("threshold")
    if threshold is not None:
        ty = y_at(float(threshold))
        pieces.extend([
            svg_line(left, ty, right, ty, stroke=THRESHOLD_COLOR, width=1.5, dash="7 5"),
            svg_text(right - 4, ty - 7, "real-time boundary (RTF = 1)", size=11,
                     anchor="end", fill=THRESHOLD_COLOR),
        ])

    median_points: dict[str, list[str]] = {item["key"]: [] for item in series}
    for series_index, item in enumerate(series):
        color = item["color"]
        group_rows: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in item["rows"]:
            group_rows[row["NFE"]].append(row)
        x_offset = (-13 if series_index == 0 else 13) if comparison else 0
        jitter_extent = 11 if comparison else 13
        denominator = max(len(all_prompt_ids) - 1, 1)
        for nfe in NFES:
            cx = x_at(nfe) + x_offset
            stats = summaries[item["key"]][nfe]
            low_y = y_at(float(stats[q1_field]))
            high_y = y_at(float(stats[q3_field]))
            median = float(stats[median_field])
            cy = y_at(median)
            pieces.extend([
                svg_line(cx, high_y, cx, low_y, stroke=color, width=2),
                svg_line(cx - 7, high_y, cx + 7, high_y, stroke=color, width=2),
                svg_line(cx - 7, low_y, cx + 7, low_y, stroke=color, width=2),
            ])
            for row in group_rows[nfe]:
                rank = prompt_rank[row["prompt_id"]]
                jitter = (rank - (len(all_prompt_ids) - 1) / 2) / denominator * jitter_extent
                pieces.append(
                    f'<circle cx="{cx + jitter:.2f}" cy="{y_at(float(row[field])):.2f}" '
                    f'r="3.2" fill="{color}" fill-opacity="0.28" stroke="#ffffff" stroke-width="0.45"/>'
                )
            median_points[item["key"]].append(f"{cx:.2f},{cy:.2f}")

    for series_index, item in enumerate(series):
        points = median_points[item["key"]]
        color = item["color"]
        pieces.append(
            f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" '
            'stroke-width="3" stroke-linejoin="round"/>'
        )
        for nfe, point in zip(NFES, points):
            cx_s, cy_s = point.split(",")
            cx, cy = float(cx_s), float(cy_s)
            stats = summaries[item["key"]][nfe]
            value = float(stats[median_field])
            pieces.append(
                f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5.5" fill="{color}" '
                'stroke="#ffffff" stroke-width="2"/>'
            )
            label_x = cx + (-6 if series_index == 0 else 6) if comparison else cx
            anchor = "end" if comparison and series_index == 0 else "start" if comparison else "middle"
            label_y = cy - 10 if (not comparison or series_index == 0) else cy + 17
            pieces.append(svg_text(label_x, label_y, fmt_value(metric, value), size=10,
                                   anchor=anchor, weight="bold", fill=color))

    legend_y = y + height - 43
    if comparison:
        for index, item in enumerate(series):
            start_x = left + 12 + index * 285
            pieces.extend([
                svg_line(start_x, legend_y - 4, start_x + 26, legend_y - 4,
                         stroke=item["color"], width=3),
                svg_text(start_x + 34, legend_y, item["legend"], size=11, fill=MUTED_COLOR),
            ])
        pieces.append(svg_text(right - 8, legend_y, "Dots: prompts. Whiskers: IQR.",
                               size=10, anchor="end", fill=MUTED_COLOR))
    else:
        pieces.extend([
            f'<circle cx="{left + 6:.2f}" cy="{legend_y - 4:.2f}" r="4" fill="{DOT_COLOR}"/>',
            svg_text(left + 17, legend_y, "one measured generation per prompt", size=11, fill=MUTED_COLOR),
            svg_line(left + 242, legend_y - 4, left + 270, legend_y - 4,
                     stroke=series[0]["color"], width=3),
            svg_text(left + 278, legend_y, "median", size=11, fill=MUTED_COLOR),
            svg_line(left + 365, legend_y - 12, left + 365, legend_y + 4,
                     stroke=series[0]["color"], width=2),
            svg_line(left + 359, legend_y - 12, left + 371, legend_y - 12,
                     stroke=series[0]["color"], width=2),
            svg_line(left + 359, legend_y + 4, left + 371, legend_y + 4,
                     stroke=series[0]["color"], width=2),
            svg_text(left + 380, legend_y, "IQR across prompts", size=11, fill=MUTED_COLOR),
        ])
    return "\n".join(pieces)


def write_detail_charts(
    output_dir: Path,
    rows: list[dict[str, Any]],
    stats: dict[int, dict[str, float | int]],
    *,
    mode: str,
    label: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series = [{"key": "xvector", "label": label, "legend": label, "color": XVECTOR_COLOR, "rows": rows}]
    all_stats = {"xvector": stats}
    for metric, config in METRICS.items():
        title = config["title"]
        body = [
            svg_text(600, 37, f"dots.tts NFE benchmark: {title}", size=23, anchor="middle", weight="bold"),
            svg_text(600, 64, "RTX 4090 x-vector full sweep. GPU model is run-owner reported; metadata records CUDA.",
                     size=13, anchor="middle", fill=MUTED_COLOR),
            chart_panel(
                series, all_stats, x=50, y=87, width=1100, height=600,
                metric=metric, title=title, subtitle=config["subtitle"],
                maximum=float(config["xvector_max"]), tick=float(config["xvector_tick"]), comparison=False,
            ),
            svg_text(600, 721, "54 prompts • 9 NFE settings • 486 measured generations • one result per prompt/NFE • warmups excluded",
                     size=12, anchor="middle", fill=MUTED_COLOR),
            svg_text(600, 748, "dots.tts-soar • CUDA • bfloat16 • Euler • seed 42 • guidance 1.2 • speaker scale 1.5",
                     size=12, anchor="middle", fill=MUTED_COLOR),
            svg_text(600, 775, "Performance and clip duration only. These plots do not score transcript adherence or speech quality.",
                     size=12, anchor="middle", fill="#9a3412"),
        ]
        filename = {"latency": "nfe_latency.svg", "rtf": "nfe_rtf.svg", "duration": "nfe_audio_duration.svg"}[metric]
        (output_dir / filename).write_text(
            svg_document(title, f"{title} across 54 prompts at nine NFE values. Dots are prompt measurements, line is median, whiskers are IQR.",
                         1200, 800, "\n".join(body)),
            encoding="utf-8",
        )

    dashboard = [
        svg_text(650, 38, "dots.tts full NFE sweep • RTX 4090 x-vector run", size=27, anchor="middle", weight="bold"),
        svg_text(650, 66, "54 prompts • 9 NFE settings • 486 measured generations • one per prompt/NFE • warmups excluded",
                 size=13, anchor="middle", fill=MUTED_COLOR),
        svg_text(650, 91, "Across the nine NFEs, median latency is lower than the original run by 27% to 35%.",
                 size=14, anchor="middle", weight="bold"),
        svg_text(650, 116, "Prompt set, seed, solver, precision, and guidance match. GPU labels are reported; only x-vector conditioning is recorded.",
                 size=12, anchor="middle", fill="#9a3412"),
    ]
    for index, (metric, config) in enumerate(METRICS.items()):
        dashboard.append(chart_panel(
            series, all_stats, x=55, y=145 + index * 400, width=1190, height=380,
            metric=metric, title=config["title"], subtitle=config["subtitle"],
            maximum=float(config["xvector_max"]), tick=float(config["xvector_tick"]), comparison=False,
        ))
    dashboard.append(svg_text(650, 1375,
                              "GPU model is reported by the run owner; archived result rows identify the device only as CUDA.",
                              size=12, anchor="middle", fill=MUTED_COLOR))
    (output_dir / "nfe_benchmark_dashboard.svg").write_text(
        svg_document("dots.tts full NFE sweep: RTX 4090 x-vector run",
                     "Three plots show latency, real-time factor, and generated audio duration across the full x-vector prompt sweep.",
                     1300, 1410, "\n".join(dashboard)),
        encoding="utf-8",
    )


def write_comparison_charts(
    output_dir: Path,
    baseline_rows: list[dict[str, Any]],
    xvector_rows: list[dict[str, Any]],
    baseline_stats: dict[int, dict[str, float | int]],
    xvector_stats: dict[int, dict[str, float | int]],
    table: list[dict[str, float | int]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    series = [
        {"key": "baseline", "label": BASELINE_LABEL, "legend": "Original run • RTX 3090", "color": BASELINE_COLOR, "rows": baseline_rows},
        {"key": "xvector", "label": XVECTOR_LABEL, "legend": "X-vector run • RTX 4090", "color": XVECTOR_COLOR, "rows": xvector_rows},
    ]
    all_stats = {"baseline": baseline_stats, "xvector": xvector_stats}
    for metric, config in METRICS.items():
        title = config["title"] + " comparison"
        body = [
            svg_text(600, 37, f"Full 54-prompt sweep: {title}", size=23, anchor="middle", weight="bold"),
            svg_text(600, 64, "Matched prompt IDs and NFE settings. GPU labels follow run-owner information; JSON records CUDA.",
                     size=13, anchor="middle", fill=MUTED_COLOR),
            chart_panel(
                series, all_stats, x=50, y=87, width=1100, height=600,
                metric=metric, title=title, subtitle=config["subtitle"],
                maximum=float(config["comparison_max"]), tick=float(config["comparison_tick"]), comparison=True,
            ),
            svg_text(600, 721, "54 matched prompts • 486 measured generations per run • one result per prompt/NFE • warmups excluded",
                     size=12, anchor="middle", fill=MUTED_COLOR),
            svg_text(600, 748, "Same NFE values, prompt snapshots, seed, solver, precision, and guidance. Only the x-vector run records voice conditioning.",
                     size=12, anchor="middle", fill=MUTED_COLOR),
            svg_text(600, 775, "Run-to-run comparison. The baseline conditioning is unspecified, so this does not isolate the x-vector effect.",
                     size=12, anchor="middle", weight="bold", fill="#9a3412"),
        ]
        filename = {"latency": "latency_comparison.svg", "rtf": "rtf_comparison.svg", "duration": "duration_comparison.svg"}[metric]
        (output_dir / filename).write_text(
            svg_document(title, f"Comparison of the original RTX 3090 run and x-vector RTX 4090 run for {metric}. Each dot is a prompt result; lines are medians and whiskers are IQR.",
                         1200, 800, "\n".join(body)),
            encoding="utf-8",
        )

    median_savings = [float(row["latency_median_change_pct"]) for row in table]
    median_rtf_changes = [float(row["rtf_median_change_pct"]) for row in table]
    median_duration_changes = [float(row["duration_median_change_s"]) for row in table]
    dashboard = [
        svg_text(650, 38, "dots.tts full NFE sweep: RTX 3090 vs RTX 4090 x-vector", size=26, anchor="middle", weight="bold"),
        svg_text(650, 66, "54 matched prompts • 9 NFE settings • one measured generation per prompt/NFE • 486 results per run",
                 size=13, anchor="middle", fill=MUTED_COLOR),
        svg_text(650, 92, f"Median latency by NFE is {min(abs(value) for value in median_savings):.0f}% to {max(abs(value) for value in median_savings):.0f}% lower in the x-vector 4090 run.",
                 size=14, anchor="middle", weight="bold"),
        svg_text(650, 117, f"Median RTF changes {min(median_rtf_changes):.0f}% to {max(median_rtf_changes):.0f}%; median WAV length changes {min(median_duration_changes):.2f} to {max(median_duration_changes):.2f} seconds.",
                 size=12, anchor="middle", fill=MUTED_COLOR),
        svg_text(650, 141, "Hardware and conditioning changed together. This compares the two full systems, not the x-vector effect alone.",
                 size=12, anchor="middle", weight="bold", fill="#9a3412"),
    ]
    for index, (metric, config) in enumerate(METRICS.items()):
        dashboard.append(chart_panel(
            series, all_stats, x=55, y=152 + index * 400, width=1190, height=380,
            metric=metric, title=config["title"], subtitle=config["subtitle"],
            maximum=float(config["comparison_max"]), tick=float(config["comparison_tick"]), comparison=True,
        ))
    dashboard.append(svg_text(650, 1383,
                              "Colors: blue = original run, orange = x-vector run. Dots = prompts. Whiskers = IQR. Warmups excluded.",
                              size=12, anchor="middle", fill=MUTED_COLOR))
    (output_dir / "comparison_dashboard.svg").write_text(
        svg_document("dots.tts NFE comparison: RTX 3090 and RTX 4090 x-vector",
                     "Matched full-sweep comparison of latency, real-time factor, and output duration by NFE.",
                     1300, 1420, "\n".join(dashboard)),
        encoding="utf-8",
    )


def write_summary_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    baseline_meta: dict[str, Any],
    baseline_prompts: dict[str, str],
    baseline_rows: list[dict[str, Any]],
    xvector_meta: dict[str, Any],
    xvector_rows: list[dict[str, Any]],
    table: list[dict[str, float | int]],
) -> None:
    graphs_from_report = Path(os.path.relpath(COMPARISON_GRAPHS, ANALYSIS_DIR)).as_posix()
    xvector_from_report = Path(os.path.relpath(XVECTOR_GRAPHS, ANALYSIS_DIR)).as_posix()
    baseline_p50_savings = [float(row["latency_median_change_pct"]) for row in table]
    paired_savings = [
        100 * (1 - float(row["paired_latency_ratio_median"])) for row in table
    ]
    rtf_changes = [float(row["rtf_median_change_pct"]) for row in table]
    duration_changes = [float(row["duration_median_change_s"]) for row in table]
    baseline_gpu = "RTX 3090, supplied by run owner"
    xvector_gpu = "RTX 4090, supplied by run owner"
    baseline_durations = [row["audio_duration_seconds"] for row in baseline_rows]
    xvector_durations = [row["audio_duration_seconds"] for row in xvector_rows]
    baseline_max_duration = max(baseline_durations)
    baseline_at_max = sum(
        duration >= baseline_max_duration - CAP_TOLERANCE_SECONDS
        for duration in baseline_durations
    )
    xvector_cap = int(xvector_meta["max_generate_length"]) * PATCH_SECONDS
    xvector_at_cap = sum(
        duration >= xvector_cap - CAP_TOLERANCE_SECONDS for duration in xvector_durations
    )
    lines = [
        "# Full NFE sweep comparison: original run and x-vector rerun",
        "",
        "> **Data validity warning.** The original run passed the reference transcript "
        "(continuation mode; `nfe_benchmark.py` before commit `65d27f5`), and "
        f"{baseline_at_max} of {len(baseline_rows)} of its WAVs stop at exactly "
        f"{baseline_max_duration:.2f} s, the 80-patch generation cap minus the reference audio. "
        f"In the x-vector run, {xvector_at_cap} of {len(xvector_rows)} WAVs stop at the "
        f"{xvector_cap:.1f} s cap. Median WAV duration is "
        f"{statistics.median(baseline_durations):.2f} s vs "
        f"{statistics.median(xvector_durations):.2f} s, so the latency comparison below mostly "
        "reflects how much audio each run generated, not the GPU or x-vector conditioning. "
        f"Per-NFE median RTF ({min(rtf_changes):.1f}% to {max(rtf_changes):.1f}%) is the fairer "
        "comparison. See `experiments/clean_samples/` for per-WAV flags and ASR-checked samples.",
        "",
        "## Scope and validation",
        "",
        "The original and x-vector datasets each contain 486 measured generations: 54 prompts multiplied by 9 NFE values. Every prompt/NFE pair appears once in each run. The two prompt snapshots match exactly, and the comparison is paired by prompt ID and NFE. All referenced WAVs exist, their sample durations match results.csv, and the 27 WAVs at each NFE in each part have distinct waveform samples. This rules out exact waveform reuse across prompts in these archived parts; it does not establish that each WAV speaks its assigned text.",
        "",
        "Both run metadata files record CUDA rather than a GPU model. The RTX 3090 label for the original run and RTX 4090 label for the x-vector run come from the run owner. The x-vector run metadata explicitly records `voice_conditioning: x-vector`; the original run metadata does not contain a voice-conditioning field.",
        "",
        f"Shared recorded settings: `{baseline_meta['model']}`, bfloat16, CUDA, optimization disabled, seed {baseline_meta['seed']}, Euler, guidance {baseline_meta['guidance_scale']}, speaker scale {baseline_meta['speaker_scale']}, max generation length {baseline_meta['max_generate_length']}, max sequence length {baseline_meta['max_sequence_length']}, one measured generation per prompt/NFE, and one unmeasured representative-prompt warmup per NFE.",
        "",
        "The prompt manifest snapshots, NFE settings, and recorded generation settings match. The reported GPU model differs, and the original run has no voice-conditioning field. Therefore, latency differences compare the two archived configurations and do not isolate the effect of x-vector conditioning.",
        "",
        "## Charts",
        "",
        f"- [Full comparison dashboard]({graphs_from_report}/comparison_dashboard.svg)",
        f"- [Latency comparison]({graphs_from_report}/latency_comparison.svg)",
        f"- [RTF comparison]({graphs_from_report}/rtf_comparison.svg)",
        f"- [Audio duration comparison]({graphs_from_report}/duration_comparison.svg)",
        f"- [Standalone x-vector 4090 dashboard]({xvector_from_report}/nfe_benchmark_dashboard.svg)",
        "",
        "Each dot is one prompt's measured generation at that NFE. The line shows the median across 54 prompts, and the whiskers show the interquartile range. The RTF chart marks 1.0, the real-time boundary. Warmups are excluded.",
        "",
        "## What changed in the measurements",
        "",
        f"- The ratio of per-NFE median latencies is {min(abs(value) for value in baseline_p50_savings):.1f}% to {max(abs(value) for value in baseline_p50_savings):.1f}% lower in the x-vector 4090 run. The median of prompt-matched latency reductions ranges from {min(paired_savings):.1f}% to {max(paired_savings):.1f}% across NFEs.",
        f"- The ratio of per-NFE median RTF values changes by {min(rtf_changes):.1f}% to {max(rtf_changes):.1f}%. An RTF below 1 means the measured generation took less time than the resulting clip's playback duration.",
        f"- Baseline median WAV duration is {min(float(row['baseline_duration_median']) for row in table):.2f} seconds at each NFE. X-vector medians range from {min(float(row['xvector_duration_median']) for row in table):.2f} to {max(float(row['xvector_duration_median']) for row in table):.2f} seconds, {min(abs(value) for value in duration_changes):.2f} to {max(abs(value) for value in duration_changes):.2f} seconds shorter by NFE median.",
        "- Output duration is not a speech-quality or transcript-adherence score. The run outputs have not been automatically evaluated for text accuracy or listened to as part of this chart generation.",
        "",
        "## Per-NFE medians",
        "",
        "| NFE | Prompts | Latency 3090 (s) | Latency 4090 x-vector (s) | Change in medians | RTF 3090 | RTF 4090 x-vector | WAV 3090 (s) | WAV 4090 x-vector (s) |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in table:
        change = float(row["latency_median_change_pct"])
        lines.append(
            f"| {row['NFE']} | {row['prompts']} | {float(row['baseline_latency_median']):.2f} | "
            f"{float(row['xvector_latency_median']):.2f} | {change:+.1f}% | "
            f"{float(row['baseline_rtf_median']):.2f} | {float(row['xvector_rtf_median']):.2f} | "
            f"{float(row['baseline_duration_median']):.2f} | {float(row['xvector_duration_median']):.2f} |"
        )
    lines.extend([
        "",
        "## Source runs",
        "",
        f"- Original full sweep, {baseline_gpu}: `nfe_54_prompts_v1_full_part_0` and `nfe_54_prompts_v1_full_part_1`.",
        f"- X-vector full sweep, {xvector_gpu}: `nfe_54_prompts_v1_full_xvector_part_0` and `nfe_54_prompts_v1_full_xvector_part_1`.",
        "- Hardware model labels are documented from the run owner's information because the archived JSON and result rows identify only `device=cuda`.",
        "",
    ])
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ANALYSIS_DIR / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    write_summary_csv(table, ANALYSIS_DIR / "comparison_by_nfe.csv")


def data_validity_lines() -> list[str]:
    """Run-index warning built from clean_samples/run_summary.json, if present."""
    if not CLEAN_SAMPLES_SUMMARY.is_file():
        return []
    summary = json.loads(CLEAN_SAMPLES_SUMMARY.read_text(encoding="utf-8"))
    baseline = summary["baseline_continuation_3090"]
    xvector = summary["xvector_4090"]
    changes = summary["xvector_vs_baseline_per_nfe"].values()
    latency_changes = [abs(change["latency_median_change_pct"]) for change in changes]
    rtf_changes = [abs(change["rtf_median_change_pct"]) for change in changes]
    clean_max = max(
        float(row["duration_seconds"])
        for row in read_csv(CLEAN_SAMPLES_SUMMARY.with_name("wav_flags.csv"))
        if row["status"] == "clean_candidate"
    )
    short_at_max = baseline["per_bucket"]["short"]["at_run_max_duration"]
    short_rows = baseline["per_bucket"]["short"]["rows"]
    xvector_cap = xvector["max_duration_seconds"]
    return [
        "## Data validity — read before using these WAVs or latency numbers",
        "",
        f"Every archived run used `max_generate_length=80`. One latent patch is {summary['patch_seconds']} s, so generation is capped at {xvector_cap:.1f} s of audio. Per-WAV flags are in [`clean_samples/wav_flags.csv`](../clean_samples/wav_flags.csv) and the figures below come from [`clean_samples/run_summary.json`](../clean_samples/run_summary.json).",
        "",
        f"- **Original run (`nfe_54_prompts_v1_full_part_*`) is continuation mode and capped.** Before commit `65d27f5`, `nfe_benchmark.py` always passed the reference transcript as `prompt_text`, so this run conditioned on \"Hey, Adam here…\" as continuation context. The reference audio consumes part of the 80-patch budget; {baseline['rows_at_run_max_duration']} of {baseline['rows']} WAVs stop at exactly {baseline['max_duration_seconds']:.2f} s, including every medium and long prompt and {short_at_max} of {short_rows} short ones. Generation length, and therefore latency, is set by the cap rather than the text. All {baseline['rows']} rows are flagged `contaminated_continuation`.",
        "- **Pilot and smoke runs (`nfe_sweep_*`, `abhi_smoke_*`, `smoke_*`) are also continuation mode** and flagged the same way.",
        f"- **X-vector run (`nfe_54_prompts_v1_full_xvector_part_*`): all {xvector['per_bucket']['long']['at_cap']} long-prompt WAVs are truncated** at {xvector_cap:.1f} s (`truncated_at_cap`). The {xvector['status_counts']['clean_candidate']} short and medium WAVs end before the cap (longest {clean_max:.2f} s) and are `clean_candidate`.",
        f"- **The original vs x-vector latency comparison is confounded.** Per-NFE median latency is {min(latency_changes):.1f}–{max(latency_changes):.1f}% lower in the x-vector run, but its median WAV is {summary['xvector_overall_duration_median_s']:.1f} s against {summary['baseline_overall_duration_median_s']:.2f} s, so it generated less audio. Per-NFE median RTF differs by only {min(rtf_changes):.1f}–{max(rtf_changes):.1f}%, which is the fairer comparison, and even that mixes conditioning mode, GPU, and the cap.",
        "- **New runs default to `max_generate_length=500`** (the upstream default) and warn when an x-vector output reaches the cap. Their latency numbers are not directly comparable to these archived runs.",
        "",
        "ASR-checked clean and demo-safe samples are in [`clean_samples/`](../clean_samples/).",
        "",
    ]


def write_indexes() -> None:
    graphs_readme = [
        "# Benchmark figures",
        "",
        "Figures are grouped by experiment so the single-prompt pilot, x-vector rerun, and full-sweep comparison are not confused.",
        "",
        "## 01. RTX 4090 single-prompt pilot",
        "",
        "- [Dashboard](01_pilot_rtx4090_single_prompt/nfe_benchmark_dashboard.svg)",
        "- [Latency](01_pilot_rtx4090_single_prompt/nfe_latency.svg)",
        "- [RTF](01_pilot_rtx4090_single_prompt/nfe_rtf.svg)",
        "- [Audio duration](01_pilot_rtx4090_single_prompt/nfe_audio_duration.svg)",
        "- [Pilot executive summary](../experiments/nfe_benchmark_runs/nfe_sweep_withwarmup_20261012/analysis/executive_summary.md)",
        "",
        "## 02. RTX 4090 x-vector full sweep",
        "",
        "- [Dashboard](02_full_rtx4090_xvector_54_prompts/nfe_benchmark_dashboard.svg)",
        "- [Latency](02_full_rtx4090_xvector_54_prompts/nfe_latency.svg)",
        "- [RTF](02_full_rtx4090_xvector_54_prompts/nfe_rtf.svg)",
        "- [Audio duration](02_full_rtx4090_xvector_54_prompts/nfe_audio_duration.svg)",
        "",
        "## 03. Full sweep comparison",
        "",
        "- [Comparison dashboard](03_full_sweep_rtx3090_vs_rtx4090/comparison_dashboard.svg)",
        "- [Latency](03_full_sweep_rtx3090_vs_rtx4090/latency_comparison.svg)",
        "- [RTF](03_full_sweep_rtx3090_vs_rtx4090/rtf_comparison.svg)",
        "- [Audio duration](03_full_sweep_rtx3090_vs_rtx4090/duration_comparison.svg)",
        "- [Comparison analysis](../experiments/nfe_benchmark_runs/analysis/full_sweep_xvector_comparison/comparison.md)",
        "",
    ]
    (ROOT / "graphs" / "README.md").write_text("\n".join(graphs_readme), encoding="utf-8")

    runs_readme = [
        "# NFE benchmark run index",
        "",
        "The full runs are split into two parts, with 27 prompts in each. Part 0 contains p001-p027; part 1 contains p028-p054. Each prompt is measured once at each of the nine NFE values, giving 486 rows per complete run.",
        "",
        *data_validity_lines(),
        "## Run groups",
        "",
        "| Run | Hardware label | Conditioning / purpose | Data folders | Figures and analysis |",
        "| --- | --- | --- | --- | --- |",
        "| Original full run | RTX 3090, run-owner reported | Original 54-prompt run | `nfe_54_prompts_v1_full_part_0`, `nfe_54_prompts_v1_full_part_1` | [Comparison report](analysis/full_sweep_xvector_comparison/comparison.md) |",
        "| X-vector rerun | RTX 4090, run-owner reported | X-vector voice conditioning | `nfe_54_prompts_v1_full_xvector_part_0`, `nfe_54_prompts_v1_full_xvector_part_1` | [X-vector charts](../../graphs/02_full_rtx4090_xvector_54_prompts/) |",
        "| Single-prompt pilot | RTX 4090, from its run summary | 1 prompt, 9 NFEs, 5 measured repeats | `nfe_sweep_withwarmup_20261012` | [Pilot analysis](nfe_sweep_withwarmup_20261012/analysis/executive_summary.md) |",
        "",
        "## Configuration and interpretation",
        "",
        "The full-sweep prompt snapshots and recorded settings match across the original and x-vector runs: dots.tts-soar, bfloat16, seed 42, Euler, guidance 1.2, speaker scale 1.5, generation length 80, sequence length 2048, and one unmeasured representative-prompt warmup per NFE. The x-vector metadata records `voice_conditioning: x-vector`; the original metadata does not record a conditioning field. GPU model is not stored in either full-run metadata; the 3090 and 4090 labels are supplied by the run owner.",
        "",
        "The reported GPU model differs, and the original run does not record its voice-conditioning setting. The comparison therefore describes the two archived runs, not a controlled estimate of x-vector's isolated effect. Audio duration and timing do not establish transcript correctness or speech quality.",
        "",
        "## Regenerate charts",
        "",
        "From the repository root, run `python experiments/plot_nfe_sweep_comparison.py`. The script validates prompt IDs, prompt text, configuration, row counts, one-row-per-prompt/NFE coverage, WAV existence and duration, and unique waveform samples per NFE within each part before writing comparison charts and summaries.",
        "",
        "See the [figure index](../../graphs/README.md) for chart links.",
        "",
    ]
    (RUN_ROOT / "README.md").write_text("\n".join(runs_readme), encoding="utf-8")


def main() -> int:
    baseline_meta, baseline_prompts, baseline_rows = load_pair(BASELINE_PARTS, "Original run")
    xvector_meta, xvector_prompts, xvector_rows = load_pair(XVECTOR_PARTS, "X-vector run")
    validate_comparable_runs(
        baseline_meta, baseline_prompts, baseline_rows,
        xvector_meta, xvector_prompts, xvector_rows,
    )
    baseline_stats = summarize(baseline_rows)
    xvector_stats = summarize(xvector_rows)
    table = comparison_summary(baseline_rows, xvector_rows, baseline_stats, xvector_stats)
    write_detail_charts(
        XVECTOR_GRAPHS, xvector_rows, xvector_stats,
        mode="xvector", label=XVECTOR_LABEL,
    )
    write_comparison_charts(
        COMPARISON_GRAPHS, baseline_rows, xvector_rows,
        baseline_stats, xvector_stats, table,
    )
    write_report(
        baseline_meta, baseline_prompts, baseline_rows,
        xvector_meta, xvector_rows, table,
    )
    write_indexes()
    print(f"Validated matched prompts: {len(baseline_prompts)}")
    print(f"Validated measured rows per full run: {len(baseline_rows)}")
    print(f"Wrote x-vector figures to {XVECTOR_GRAPHS}")
    print(f"Wrote comparison figures to {COMPARISON_GRAPHS}")
    print(f"Wrote analysis to {ANALYSIS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
