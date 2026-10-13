# NFE benchmark run index

The full runs are split into two parts, with 27 prompts in each. Part 0 contains p001-p027; part 1 contains p028-p054. Each prompt is measured once at each of the nine NFE values, giving 486 rows per complete run.

## Run groups

| Run | Hardware label | Conditioning / purpose | Data folders | Figures and analysis |
| --- | --- | --- | --- | --- |
| Original full run | RTX 3090, run-owner reported | Original 54-prompt run | `nfe_54_prompts_v1_full_part_0`, `nfe_54_prompts_v1_full_part_1` | [Comparison report](analysis/full_sweep_xvector_comparison/comparison.md) |
| X-vector rerun | RTX 4090, run-owner reported | X-vector voice conditioning | `nfe_54_prompts_v1_full_xvector_part_0`, `nfe_54_prompts_v1_full_xvector_part_1` | [X-vector charts](../../graphs/02_full_rtx4090_xvector_54_prompts/) |
| Single-prompt pilot | RTX 4090, from its run summary | 1 prompt, 9 NFEs, 5 measured repeats | `nfe_sweep_withwarmup_20261012` | [Pilot analysis](nfe_sweep_withwarmup_20261012/analysis/executive_summary.md) |

## Configuration and interpretation

The full-sweep prompt snapshots and recorded settings match across the original and x-vector runs: dots.tts-soar, bfloat16, seed 42, Euler, guidance 1.2, speaker scale 1.5, generation length 80, sequence length 2048, and one unmeasured representative-prompt warmup per NFE. The x-vector metadata records `voice_conditioning: x-vector`; the original metadata does not record a conditioning field. GPU model is not stored in either full-run metadata; the 3090 and 4090 labels are supplied by the run owner.

The reported GPU model differs, and the original run does not record its voice-conditioning setting. The comparison therefore describes the two archived runs, not a controlled estimate of x-vector's isolated effect. Audio duration and timing do not establish transcript correctness or speech quality.

## Regenerate charts

From the repository root, run `python experiments/plot_nfe_sweep_comparison.py`. The script validates prompt IDs, prompt text, configuration, row counts, one-row-per-prompt/NFE coverage, WAV existence and duration, and unique waveform samples per NFE within each part before writing comparison charts and summaries.

See the [figure index](../../graphs/README.md) for chart links.
