# Full NFE sweep comparison: original run and x-vector rerun

## Scope and validation

The original and x-vector datasets each contain 486 measured generations: 54 prompts multiplied by 9 NFE values. Every prompt/NFE pair appears once in each run. The two prompt snapshots match exactly, and the comparison is paired by prompt ID and NFE. All referenced WAVs exist, their sample durations match results.csv, and the 27 WAVs at each NFE in each part have distinct waveform samples. This rules out exact waveform reuse across prompts in these archived parts; it does not establish that each WAV speaks its assigned text.

Both run metadata files record CUDA rather than a GPU model. The RTX 3090 label for the original run and RTX 4090 label for the x-vector run come from the run owner. The x-vector run metadata explicitly records `voice_conditioning: x-vector`; the original run metadata does not contain a voice-conditioning field.

Shared recorded settings: `dots-studio/dots.tts-soar`, bfloat16, CUDA, optimization disabled, seed 42, Euler, guidance 1.2, speaker scale 1.5, max generation length 80, max sequence length 2048, one measured generation per prompt/NFE, and one unmeasured representative-prompt warmup per NFE.

The prompt manifest snapshots, NFE settings, and recorded generation settings match. The reported GPU model differs, and the original run has no voice-conditioning field. Therefore, latency differences compare the two archived configurations and do not isolate the effect of x-vector conditioning.

## Charts

- [Full comparison dashboard](../../../../graphs/03_full_sweep_rtx3090_vs_rtx4090/comparison_dashboard.svg)
- [Latency comparison](../../../../graphs/03_full_sweep_rtx3090_vs_rtx4090/latency_comparison.svg)
- [RTF comparison](../../../../graphs/03_full_sweep_rtx3090_vs_rtx4090/rtf_comparison.svg)
- [Audio duration comparison](../../../../graphs/03_full_sweep_rtx3090_vs_rtx4090/duration_comparison.svg)
- [Standalone x-vector 4090 dashboard](../../../../graphs/02_full_rtx4090_xvector_54_prompts/nfe_benchmark_dashboard.svg)

Each dot is one prompt's measured generation at that NFE. The line shows the median across 54 prompts, and the whiskers show the interquartile range. The RTF chart marks 1.0, the real-time boundary. Warmups are excluded.

## What changed in the measurements

- The ratio of per-NFE median latencies is 27.4% to 35.3% lower in the x-vector 4090 run. The median of prompt-matched latency reductions ranges from 32.9% to 48.5% across NFEs.
- The ratio of per-NFE median RTF values changes by -10.9% to -6.2%. An RTF below 1 means the measured generation took less time than the resulting clip's playback duration.
- Baseline median WAV duration is 9.12 seconds at each NFE. X-vector medians range from 6.40 to 7.12 seconds, 2.00 to 2.72 seconds shorter by NFE median.
- Output duration is not a speech-quality or transcript-adherence score. The run outputs have not been automatically evaluated for text accuracy or listened to as part of this chart generation.

## Per-NFE medians

| NFE | Prompts | Latency 3090 (s) | Latency 4090 x-vector (s) | Change in medians | RTF 3090 | RTF 4090 x-vector | WAV 3090 (s) | WAV 4090 x-vector (s) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 54 | 7.67 | 5.53 | -27.9% | 0.84 | 0.79 | 9.12 | 7.12 |
| 6 | 54 | 9.87 | 7.17 | -27.4% | 1.08 | 1.00 | 9.12 | 6.96 |
| 8 | 54 | 12.15 | 8.56 | -29.6% | 1.33 | 1.24 | 9.12 | 6.96 |
| 10 | 54 | 14.30 | 9.96 | -30.3% | 1.57 | 1.47 | 9.12 | 6.72 |
| 12 | 54 | 17.10 | 11.07 | -35.3% | 1.87 | 1.73 | 9.12 | 6.48 |
| 16 | 54 | 21.77 | 14.50 | -33.4% | 2.39 | 2.22 | 9.12 | 6.72 |
| 20 | 54 | 27.11 | 17.81 | -34.3% | 2.97 | 2.65 | 9.12 | 6.64 |
| 24 | 54 | 30.55 | 21.10 | -30.9% | 3.37 | 3.14 | 9.12 | 6.72 |
| 32 | 54 | 39.71 | 26.21 | -34.0% | 4.38 | 4.06 | 9.12 | 6.40 |

## Source runs

- Original full sweep, RTX 3090, supplied by run owner: `nfe_54_prompts_v1_full_part_0` and `nfe_54_prompts_v1_full_part_1`.
- X-vector full sweep, RTX 4090, supplied by run owner: `nfe_54_prompts_v1_full_xvector_part_0` and `nfe_54_prompts_v1_full_xvector_part_1`.
- Hardware model labels are documented from the run owner's information because the archived JSON and result rows identify only `device=cuda`.
