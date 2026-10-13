# dots.tts NFE benchmark: executive summary

**Scope:** one target prompt, `short`: “The small red bird landed on the garden fence.”
**Coverage:** 45 measured generations across 9 NFE settings, 5 repeats per setting. Warmups are excluded.
**Configuration:** RTX 4090, CUDA, bfloat16, Euler, seed 42, guidance 1.2, speaker scale 1.5.

## Findings

- Median latency rises from 6.96 seconds at NFE 4 to 36.09 seconds at NFE 32, a 5.19x increase.
- Median RTF is below 1 only at NFE 4. NFE 6 is near parity at 1.01.
- Generated audio duration ranges from 8.16 to 9.12 seconds.

> **Interpretation limit:** this is a performance trace for one prompt. The earlier run was reported to repeat the reference phrase, so prompt adherence and speech quality remain unverified. Do not present the plots as a 54-prompt quality evaluation.

## Figures and details

- [Full dashboard](../../../../graphs/01_pilot_rtx4090_single_prompt/nfe_benchmark_dashboard.svg)
- [Latency detail](../../../../graphs/01_pilot_rtx4090_single_prompt/nfe_latency.svg)
- [Real-time factor detail](../../../../graphs/01_pilot_rtx4090_single_prompt/nfe_rtf.svg)
- [Audio duration detail](../../../../graphs/01_pilot_rtx4090_single_prompt/nfe_audio_duration.svg)
- [Full analysis](analysis.md)
- [Per-NFE summary table](nfe_summary.csv)
