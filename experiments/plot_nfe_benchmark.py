"""Create detailed, dependency-free SVG charts from NFE benchmark results.

Example:
    python experiments/plot_nfe_benchmark.py \
        experiments/nfe_benchmark_runs/nfe_sweep_withwarmup_20261012/results.csv
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


NFE_COLOR = "#2563eb"
DOT_COLOR = "#94a3b8"
MEDIAN_COLOR = "#0f172a"
GRID_COLOR = "#e2e8f0"
TEXT_COLOR = "#1e293b"
MUTED_COLOR = "#64748b"
THRESHOLD_COLOR = "#dc2626"


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def load_rows(path: Path, prompt_id: str | None) -> tuple[list[dict[str, Any]], str]:
    with path.open("r", newline="", encoding="utf-8-sig") as results_file:
        rows = list(csv.DictReader(results_file))
    if not rows:
        raise ValueError(f"No benchmark rows found in {path}")

    available_ids = sorted({row["prompt_id"] for row in rows})
    if prompt_id is not None:
        rows = [row for row in rows if row["prompt_id"] == prompt_id]
        if not rows:
            raise ValueError(
                f"Prompt {prompt_id!r} does not occur in {path}; "
                f"available prompt IDs: {available_ids}"
            )
    elif len(available_ids) != 1:
        raise ValueError(
            "This report is for a single target prompt. Select one with "
            f"--prompt-id; available IDs: {available_ids}"
        )

    for row in rows:
        row["NFE"] = int(row["NFE"])
        row["repeat"] = int(row["repeat"])
        for field in ("latency_seconds", "audio_duration_seconds", "RTF"):
            row[field] = float(row[field])

    return rows, rows[0]["prompt_id"]


def load_prompt_text(source: Path, prompt_id: str) -> str:
    prompt_path = source.parent / "prompts.csv"
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Prompt manifest is required for chart labels: {prompt_path}")
    with prompt_path.open("r", newline="", encoding="utf-8-sig") as prompt_file:
        prompt_text = next(
            (row.get("text", "") for row in csv.DictReader(prompt_file)
             if row.get("prompt_id") == prompt_id),
            "",
        )
    if not prompt_text:
        raise ValueError(f"Prompt ID {prompt_id!r} was not found in {prompt_path}")
    return prompt_text


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["NFE"]].append(row)

    summaries: list[dict[str, float | int]] = []
    for nfe, group in sorted(groups.items()):
        latency = [row["latency_seconds"] for row in group]
        duration = [row["audio_duration_seconds"] for row in group]
        rtf = [row["RTF"] for row in group]
        summaries.append(
            {
                "NFE": nfe,
                "repeats": len(group),
                "latency_min_s": min(latency),
                "latency_q1_s": percentile(latency, 0.25),
                "latency_median_s": statistics.median(latency),
                "latency_q3_s": percentile(latency, 0.75),
                "latency_max_s": max(latency),
                "latency_mean_s": statistics.mean(latency),
                "latency_sd_s": statistics.stdev(latency) if len(latency) > 1 else 0.0,
                "duration_median_s": statistics.median(duration),
                "duration_min_s": min(duration),
                "duration_max_s": max(duration),
                "rtf_min": min(rtf),
                "rtf_q1": percentile(rtf, 0.25),
                "rtf_median": statistics.median(rtf),
                "rtf_q3": percentile(rtf, 0.75),
                "rtf_max": max(rtf),
                "latency_vs_nfe4": 0.0,
            }
        )

    baseline = float(summaries[0]["latency_median_s"])
    for item in summaries:
        item["latency_vs_nfe4"] = float(item["latency_median_s"]) / baseline
    return summaries


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
        f'y2="{y2:.2f}" stroke="{stroke}" stroke-width="{width}"'
        f'{dash_attr}/>'
    )


def chart_panel(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, float | int]],
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    metric: str,
    title: str,
    subtitle: str,
    ylabel: str,
    maximum: float,
    tick: float,
    minimum: float = 0.0,
    threshold: float | None = None,
) -> str:
    left = x + 82
    right = x + width - 28
    top = y + 70
    bottom = y + height - 78
    plot_width = right - left
    plot_height = bottom - top
    nfes = [int(item["NFE"]) for item in summaries]
    nfe_min, nfe_max = min(nfes), max(nfes)
    x_at = lambda nfe: left + (nfe - nfe_min) / (nfe_max - nfe_min) * plot_width
    y_at = lambda value: bottom - ((value - minimum) / (maximum - minimum)) * plot_height
    pieces = [
        f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" rx="12" fill="#ffffff" '
        f'stroke="#cbd5e1" stroke-width="1"/>',
        svg_text(x + 24, y + 30, title, size=21, weight="bold"),
        svg_text(x + 24, y + 52, subtitle, size=13, fill=MUTED_COLOR),
    ]

    tick_count = math.floor((maximum - minimum) / tick)
    for tick_index in range(tick_count + 1):
        value = minimum + tick_index * tick
        cy = y_at(value)
        pieces.append(svg_line(left, cy, right, cy))
        label = f"{value:g}" if metric == "rtf" else f"{value:g}"
        pieces.append(svg_text(left - 12, cy + 5, label, size=12, anchor="end", fill=MUTED_COLOR))

    pieces.extend(
        [
            svg_line(left, top, left, bottom, stroke="#475569", width=1.5),
            svg_line(left, bottom, right, bottom, stroke="#475569", width=1.5),
            svg_text(x + 22, (top + bottom) / 2, ylabel, size=13, anchor="middle", fill=TEXT_COLOR,
                     extra=f'transform="rotate(-90 {x + 22:.2f} {(top + bottom) / 2:.2f})"'),
            svg_text((left + right) / 2, y + height - 18, "NFE (Euler steps)", size=13, anchor="middle"),
        ]
    )

    if threshold is not None:
        ty = y_at(threshold)
        pieces.append(svg_line(left, ty, right, ty, stroke=THRESHOLD_COLOR, width=1.5, dash="7 5"))
        pieces.append(
            svg_text(right - 4, ty - 7, "real-time boundary (RTF = 1)", size=11,
                     anchor="end", fill=THRESHOLD_COLOR)
        )

    by_nfe: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_nfe[row["NFE"]].append(row)
    for item in summaries:
        nfe = int(item["NFE"])
        cx = x_at(nfe)
        pieces.append(svg_line(cx, bottom, cx, bottom + 5, stroke="#475569", width=1))
        pieces.append(svg_text(cx, bottom + 23, str(nfe), size=12, anchor="middle", fill=MUTED_COLOR))

        if metric == "latency":
            low = float(item["latency_q1_s"])
            high = float(item["latency_q3_s"])
            median = float(item["latency_median_s"])
            values = [(row["latency_seconds"], row["repeat"]) for row in by_nfe[nfe]]
        elif metric == "rtf":
            low = float(item["rtf_q1"])
            high = float(item["rtf_q3"])
            median = float(item["rtf_median"])
            values = [(row["RTF"], row["repeat"]) for row in by_nfe[nfe]]
        else:
            durations = [row["audio_duration_seconds"] for row in by_nfe[nfe]]
            low = percentile(durations, 0.25)
            high = percentile(durations, 0.75)
            median = statistics.median(durations)
            values = [(row["audio_duration_seconds"], row["repeat"]) for row in by_nfe[nfe]]

        low_y, high_y = y_at(low), y_at(high)
        pieces.extend(
            [
                svg_line(cx, high_y, cx, low_y, stroke=NFE_COLOR, width=2),
                svg_line(cx - 8, high_y, cx + 8, high_y, stroke=NFE_COLOR, width=2),
                svg_line(cx - 8, low_y, cx + 8, low_y, stroke=NFE_COLOR, width=2),
            ]
        )
        for value, repeat in values:
            jitter = (repeat - (len(values) + 1) / 2) * 7
            pieces.append(
                f'<circle cx="{cx + jitter:.2f}" cy="{y_at(float(value)):.2f}" '
                f'r="4.5" fill="{DOT_COLOR}" fill-opacity="0.85" '
                f'stroke="#ffffff" stroke-width="1"/>'
            )

    median_points: list[str] = []
    for item in summaries:
        nfe = int(item["NFE"])
        value = (
            float(item["latency_median_s"])
            if metric == "latency"
            else float(item["rtf_median"])
            if metric == "rtf"
            else float(item["duration_median_s"])
        )
        median_points.append(f"{x_at(nfe):.2f},{y_at(value):.2f}")
    pieces.append(
        f'<polyline points="{" ".join(median_points)}" fill="none" '
        f'stroke="{NFE_COLOR}" stroke-width="3" stroke-linejoin="round"/>'
    )
    for item in summaries:
        nfe = int(item["NFE"])
        cx = x_at(nfe)
        value = (
            float(item["latency_median_s"])
            if metric == "latency"
            else float(item["rtf_median"])
            if metric == "rtf"
            else float(item["duration_median_s"])
        )
        cy = y_at(value)
        pieces.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="6" fill="{NFE_COLOR}" '
            f'stroke="#ffffff" stroke-width="2"/>'
        )
        label = f"{value:.2f}s" if metric != "rtf" else f"{value:.2f}"
        pieces.append(
            svg_text(cx, cy - 11, label, size=11, anchor="middle", weight="bold", fill=NFE_COLOR)
        )

    legend_y = y + height - 43
    pieces.extend(
        [
            f'<circle cx="{left + 5:.2f}" cy="{legend_y - 4:.2f}" r="4.5" fill="{DOT_COLOR}"/>',
            svg_text(left + 16, legend_y, "individual measured repeat", size=11, fill=MUTED_COLOR),
            svg_line(left + 188, legend_y - 4, left + 216, legend_y - 4, stroke=NFE_COLOR, width=3),
            svg_text(left + 224, legend_y, "median", size=11, fill=MUTED_COLOR),
            svg_line(left + 320, legend_y - 12, left + 320, legend_y + 4, stroke=NFE_COLOR, width=2),
            svg_line(left + 313, legend_y - 12, left + 327, legend_y - 12, stroke=NFE_COLOR, width=2),
            svg_line(left + 313, legend_y + 4, left + 327, legend_y + 4, stroke=NFE_COLOR, width=2),
            svg_text(left + 341, legend_y, "IQR (25th to 75th percentile)", size=11, fill=MUTED_COLOR),
        ]
    )
    return "\n".join(pieces)


def svg_document(title: str, width: int, height: int, body: str) -> str:
    safe_title = html.escape(title)
    description = html.escape(
        "Single-prompt dots.tts NFE timing charts. Individual points show measured "
        "repeats, the blue line shows the median, and whiskers show the interquartile range. "
        "Warmups are excluded. Timing results do not establish prompt adherence or speech quality."
    )
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<title>{safe_title}</title>
<desc>{description}</desc>
<rect width="100%" height="100%" fill="#f8fafc"/>
<style>text{{font-family:Arial,Helvetica,sans-serif}}</style>
{body}
</svg>
'''


def write_charts(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, float | int]],
    graphs_dir: Path,
    prompt_id: str,
    prompt_text: str,
) -> None:
    durations = [row["audio_duration_seconds"] for row in rows]
    duration_range = f"{min(durations):.2f} to {max(durations):.2f} seconds"
    panels = [
        {
            "metric": "latency",
            "title": "NFE vs generation latency",
            "subtitle": "Wall-clock latency per measured generation. Dots show all repeats; line shows the median.",
            "ylabel": "Latency (seconds)",
            "maximum": 40.0,
            "tick": 5.0,
            "file": "nfe_latency.svg",
        },
        {
            "metric": "rtf",
            "title": "NFE vs real-time factor",
            "subtitle": "RTF = generation latency / generated audio duration. Below 1.0 means faster than playback.",
            "ylabel": "Real-time factor (RTF)",
            "maximum": 4.5,
            "tick": 0.5,
            "threshold": 1.0,
            "file": "nfe_rtf.svg",
        },
        {
            "metric": "duration",
            "title": "NFE vs generated audio duration",
            "subtitle": f"Zoomed axis for detail. Observed duration: {duration_range}; axis starts at 7.5 seconds.",
            "ylabel": "Audio duration (seconds, zoomed)",
            "minimum": 7.5,
            "maximum": 9.5,
            "tick": 0.25,
            "file": "nfe_audio_duration.svg",
        },
    ]

    for config in panels:
        body = [
            svg_text(600, 38, f"dots.tts NFE benchmark: {config['title']}", size=23, anchor="middle", weight="bold"),
            svg_text(600, 65, f"Prompt {prompt_id}: {prompt_text}", size=14,
                     anchor="middle", fill=MUTED_COLOR),
            chart_panel(
                rows, summaries,
                x=50, y=90, width=1100, height=600,
                metric=config["metric"], title=config["title"],
                subtitle=config["subtitle"], ylabel=config["ylabel"],
                maximum=config["maximum"], tick=config["tick"],
                minimum=config.get("minimum", 0.0),
                threshold=config.get("threshold"),
            ),
            svg_text(600, 735, "dots.tts-soar • RTX 4090 • CUDA • bfloat16 • Euler • seed 42 • guidance 1.2 • speaker scale 1.5 • max generation length 80",
                     size=12, anchor="middle", fill=MUTED_COLOR),
            svg_text(600, 758, "Timing only. One target prompt. Prompt adherence and speech quality are unverified.",
                     size=12, anchor="middle", weight="bold", fill="#9a3412"),
        ]
        (graphs_dir / config["file"]).write_text(
            svg_document(config["title"], 1200, 790, "\n".join(body)), encoding="utf-8"
        )

    dashboard_parts = [
        svg_text(650, 38, "dots.tts NFE timing profile", size=28, anchor="middle", weight="bold"),
        svg_text(650, 65, f"One target prompt ({prompt_id}): {prompt_text}",
                 size=14, anchor="middle", fill=MUTED_COLOR),
        svg_text(650, 89, f"{len(rows)} measured generations across {len(summaries)} fixed NFE settings, "
                 f"{summaries[0]['repeats']} repeats per setting; warmups excluded",
                 size=13, anchor="middle", fill=MUTED_COLOR),
        svg_text(650, 112, "Median latency: "
                 f"{float(summaries[0]['latency_median_s']):.2f}s at NFE {summaries[0]['NFE']} to "
                 f"{float(summaries[-1]['latency_median_s']):.2f}s at NFE {summaries[-1]['NFE']} "
                 f"({float(summaries[-1]['latency_median_s']) / float(summaries[0]['latency_median_s']):.2f}x). "
                 f"Only NFE {', '.join(str(item['NFE']) for item in summaries if float(item['rtf_median']) < 1.0) or 'none'} "
                 "was below real time by median RTF.",
                 size=13, anchor="middle", weight="bold"),
        svg_text(650, 136, "TIMING ONLY. The prior run was reported to repeat the reference phrase. Prompt adherence remains unverified.",
                 size=12, anchor="middle", weight="bold", fill="#9a3412"),
    ]
    for index, config in enumerate(panels):
        dashboard_parts.append(
            chart_panel(
                rows, summaries,
                x=55, y=155 + index * 400, width=1190, height=380,
                metric=config["metric"], title=config["title"],
                subtitle=config["subtitle"], ylabel=config["ylabel"],
                maximum=config["maximum"], tick=config["tick"],
                minimum=config.get("minimum", 0.0),
                threshold=config.get("threshold"),
            )
        )
    dashboard_parts.append(
        svg_text(650, 1385, "Dots: individual repeats. Blue line: median. Whiskers: interquartile range. Warmups are excluded.",
                 size=13, anchor="middle", fill=MUTED_COLOR)
    )
    (graphs_dir / "nfe_benchmark_dashboard.svg").write_text(
        svg_document("dots.tts NFE timing profile", 1300, 1420, "\n".join(dashboard_parts)),
        encoding="utf-8",
    )


def write_summary_csv(summaries: list[dict[str, float | int]], path: Path) -> None:
    fields = list(summaries[0])
    with path.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(summaries)


def write_report(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, float | int]],
    *,
    source: Path,
    prompt_id: str,
    prompt_text: str,
    graphs_dir: Path,
    out_path: Path,
) -> None:
    graphs_link = Path(os.path.relpath(graphs_dir, out_path.parent)).as_posix()
    source_link = Path(os.path.relpath(source, out_path.parent)).as_posix()
    first = summaries[0]
    last = summaries[-1]
    rtf_ready = [item for item in summaries if float(item["rtf_median"]) < 1.0]
    max_duration = max(float(row["audio_duration_seconds"]) for row in rows)
    min_duration = min(float(row["audio_duration_seconds"]) for row in rows)
    latency_factor = float(last["latency_median_s"]) / float(first["latency_median_s"])
    x_values = [float(item["NFE"]) for item in summaries]
    y_values = [float(item["latency_median_s"]) for item in summaries]
    x_mean = statistics.mean(x_values)
    y_mean = statistics.mean(y_values)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values)) / sum(
        (x - x_mean) ** 2 for x in x_values
    )
    intercept = y_mean - slope * x_mean
    residual_sum_squares = sum(
        (y - (intercept + slope * x)) ** 2 for x, y in zip(x_values, y_values)
    )
    total_sum_squares = sum((y - y_mean) ** 2 for y in y_values)
    fit_r_squared = 1 - residual_sum_squares / total_sum_squares
    max_duration_count = sum(
        math.isclose(float(row["audio_duration_seconds"]), max_duration)
        for row in rows
    )
    lines = [
        "# Fixed-NFE dots.tts benchmark analysis",
        "",
        f"Source: [`results.csv`]({source_link})",
        f"Prompt ID: `{prompt_id}`",
        f"Target text: {prompt_text or '(not found in prompts.csv)'}",
        f"Measured generations: {len(rows)} across {len(summaries)} NFE values, "
        f"with {summaries[0]['repeats']} measured repeats per NFE.",
        "",
        "## Benchmark configuration",
        "",
        "The saved result rows report `dots-studio/dots.tts-soar`, CUDA, bfloat16, "
        "Euler, seed 42, guidance scale 1.2, speaker scale 1.5, max generation "
        "length 80, max sequence length 2048, and optimization disabled. The "
        "associated run summary reports one unmeasured warmup per NFE. Warmups "
        "are not present in the measured results CSV.",
        "",
        "This is one target prompt repeated across NFE settings. It does not estimate "
        "performance across the 54-prompt manifest. The earlier sweep was reported "
        "to repeatedly speak the shared reference phrase. These timing results are "
        "a single-prompt performance trace, not evidence of correct prompt adherence "
        "or speech quality.",
        "",
        "## Charts",
        "",
        f"- [Full dashboard]({graphs_link}/nfe_benchmark_dashboard.svg)",
        f"- [NFE vs latency]({graphs_link}/nfe_latency.svg)",
        f"- [NFE vs real-time factor]({graphs_link}/nfe_rtf.svg)",
        f"- [NFE vs generated audio duration]({graphs_link}/nfe_audio_duration.svg)",
        "",
        "## Behavior classification",
        "",
        f"- **Latency rises strongly with NFE:** median latency grows from "
        f"{float(first['latency_median_s']):.2f}s at NFE {first['NFE']} to "
        f"{float(last['latency_median_s']):.2f}s at NFE {last['NFE']}, a "
        f"{latency_factor:.2f}x increase. The least-squares trend across the "
        f"per-NFE medians is {slope:.3f} seconds per additional NFE with "
        f"R² {fit_r_squared:.3f}, close to linear for this prompt and configuration.",
        f"- **Real-time boundary:** median RTF is below 1 only at "
        f"{', '.join(str(item['NFE']) for item in rtf_ready) or 'none'} NFE. "
        f"NFE 6 is close to parity at {float(summaries[1]['rtf_median']):.2f}; "
        f"NFE 32 is {float(last['rtf_median']):.2f}x real time.",
        f"- **Local plateau:** median latency is effectively unchanged between "
        f"NFE {summaries[3]['NFE']} and {summaries[4]['NFE']} "
        f"({float(summaries[3]['latency_median_s']):.3f}s vs "
        f"{float(summaries[4]['latency_median_s']):.3f}s), while median audio "
        f"duration falls from {float(summaries[3]['duration_median_s']):.2f}s "
        f"to {float(summaries[4]['duration_median_s']):.2f}s.",
        f"- **Output duration is tightly clustered:** observed duration "
        f"range is {min_duration:.2f}s to {max_duration:.2f}s for target text "
        f"{prompt_text!r}. The maximum duration occurs in "
        f"{max_duration_count}/{len(rows)} measured rows. This concentration "
        "warrants checking generation termination and listening to the WAVs, "
        "especially given the reported reference-phrase repetition.",
        "- **Repeat variability:** NFE 8 and 10 have the widest latency spread; "
        "other settings are comparatively stable across the five recorded runs. "
        "With five repeats and one prompt, these spreads describe this run only.",
        "",
        "## Median results",
        "",
        "| NFE | Runs | Median latency (s) | IQR latency (s) | Median RTF | Median audio (s) | Latency / NFE 4 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in summaries:
        lines.append(
            f"| {item['NFE']} | {item['repeats']} | {float(item['latency_median_s']):.3f} | "
            f"{float(item['latency_q1_s']):.3f}–{float(item['latency_q3_s']):.3f} | "
            f"{float(item['rtf_median']):.3f} | {float(item['duration_median_s']):.2f} | "
            f"{float(item['latency_vs_nfe4']):.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Limits",
            "",
            "No new GPU generations were started while preparing this report. These "
            "plots use the existing five-repeat single-prompt run. The previous "
            "full 54-prompt sweep is not pooled into this analysis, and no claim is "
            "made that the audio content follows the target text. The current local "
            "environment has no `nvidia-smi`, and Matplotlib/NumPy are not installed. "
            "The SVGs and statistics are generated by the included Python standard-library script.",
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_executive_summary(
    rows: list[dict[str, Any]],
    summaries: list[dict[str, float | int]],
    *,
    prompt_id: str,
    prompt_text: str,
    graphs_dir: Path,
    out_path: Path,
) -> None:
    graph_link = Path(os.path.relpath(graphs_dir, out_path.parent)).as_posix()
    first, last = summaries[0], summaries[-1]
    rtf_ready = [item for item in summaries if float(item["rtf_median"]) < 1.0]
    latency_ratio = float(last["latency_median_s"]) / float(first["latency_median_s"])
    lines = [
        "# dots.tts NFE benchmark: executive summary",
        "",
        f"**Scope:** one target prompt, `{prompt_id}`: “{prompt_text}”",
        f"**Coverage:** {len(rows)} measured generations across {len(summaries)} NFE settings, "
        f"{summaries[0]['repeats']} repeats per setting. Warmups are excluded.",
        "**Configuration:** RTX 4090, CUDA, bfloat16, Euler, seed 42, guidance 1.2, speaker scale 1.5.",
        "",
        "## Findings",
        "",
        f"- Median latency rises from {float(first['latency_median_s']):.2f} seconds at "
        f"NFE {first['NFE']} to {float(last['latency_median_s']):.2f} seconds at "
        f"NFE {last['NFE']}, a {latency_ratio:.2f}x increase.",
        f"- Median RTF is below 1 only at NFE {', '.join(str(item['NFE']) for item in rtf_ready) or 'none'}. "
        f"NFE 6 is near parity at {float(summaries[1]['rtf_median']):.2f}.",
        f"- Generated audio duration ranges from "
        f"{min(float(row['audio_duration_seconds']) for row in rows):.2f} to "
        f"{max(float(row['audio_duration_seconds']) for row in rows):.2f} seconds.",
        "",
        "> **Interpretation limit:** this is a performance trace for one prompt. The earlier run was reported to repeat the reference phrase, so prompt adherence and speech quality remain unverified. Do not present the plots as a 54-prompt quality evaluation.",
        "",
        "## Figures and details",
        "",
        f"- [Full dashboard]({graph_link}/nfe_benchmark_dashboard.svg)",
        f"- [Latency detail]({graph_link}/nfe_latency.svg)",
        f"- [Real-time factor detail]({graph_link}/nfe_rtf.svg)",
        f"- [Audio duration detail]({graph_link}/nfe_audio_duration.svg)",
        "- [Full analysis](analysis.md)",
        "- [Per-NFE summary table](nfe_summary.csv)",
        "",
    ]
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--prompt-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--graphs-dir", type=Path, default=None)
    args = parser.parse_args()

    source = args.results_csv.expanduser().resolve()
    rows, prompt_id = load_rows(source, args.prompt_id)
    prompt_text = load_prompt_text(source, prompt_id)
    summaries = summarize(rows)
    out_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else source.parent / "analysis"
    )
    graphs_dir = (
        args.graphs_dir.expanduser().resolve()
        if args.graphs_dir is not None
        else Path(__file__).resolve().parents[1] / "graphs" / "01_pilot_rtx4090_single_prompt"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    write_charts(rows, summaries, graphs_dir, prompt_id, prompt_text)
    write_summary_csv(summaries, out_dir / "nfe_summary.csv")
    write_report(
        rows,
        summaries,
        source=source,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        graphs_dir=graphs_dir,
        out_path=out_dir / "analysis.md",
    )
    write_executive_summary(
        rows,
        summaries,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        graphs_dir=graphs_dir,
        out_path=out_dir / "executive_summary.md",
    )
    print(f"Analyzed {len(rows)} measured generations for prompt {prompt_id!r}.")
    print(f"Wrote charts to {graphs_dir} and analysis to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
