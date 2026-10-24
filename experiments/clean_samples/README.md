# Clean output samples (Task #8)

Trustworthy dots.tts WAVs for NFE comparisons, ASR/speaker scoring, and the live demo. Everything here is selected from the archived **RTX 4090 x-vector run** (`nfe_54_prompts_v1_full_xvector_part_0/1`); nothing was regenerated. The 4090 label comes from the run owner; run metadata records only `cuda`.

## What's here

| Path | What it is | Who uses it |
| --- | --- | --- |
| [`demo_set/`](demo_set/) | 8 prompts × NFE 4/20/32 (24 WAVs) that are safe to play for an audience, plus a listening checklist | Sai (#1), demo |
| [`safe_demo_prompts.json`](safe_demo_prompts.json) | 25 prompts with zero word errors at NFE 20 and clean signal at NFE 20/32 | Sai (#1), Rababb (#2) |
| [`clean_set/`](clean_set/) | 12 prompts (1 short + 1 medium per category) × NFE 4/20/32 (36 WAVs) with `manifest.json`/`.csv` | Arush (#9), Atharv (#10), comparisons |
| [`asr_scores.csv`](asr_scores.csv) | Transcript, WER, CER for 108 x-vector WAVs and 54 continuation WAVs | Arush (#9) |
| [`wav_flags.csv`](wav_flags.csv) | Status and signal checks for all 1,028 archived WAVs | everyone |
| [`run_summary.json`](run_summary.json) | Per-NFE and per-bucket medians for the two full runs | reports |

## Flags in `wav_flags.csv`

- `contaminated_continuation`: the run passed the reference transcript ("Hey, Adam here…") as continuation context. Whisper hears the reference intro in **54/54** of these WAVs at NFE 20, and some say only the intro. Never use these as samples.
- `truncated_at_cap`: x-vector WAV stopped at `max_generate_length=80` (12.8 s), usually mid-sentence. All 162 long-prompt WAVs.
- `clean_candidate`: x-vector WAV that ended before the cap (324 short and medium WAVs; longest 10.72 s).

`signal_ok` additionally requires no clipping, peak ≤ 0.95, leading silence < 0.5 s, trailing silence ≥ 0.05 s, and no internal gap > 1.0 s.

## Selection rule

- ASR: `openai/whisper-large-v3`, greedy, English, Whisper's English text normalizer on both sides. On Apple MPS the script uses eager attention, because SDPA decodes every clip to `!!!!`.
- A WAV passes when word errors ≤ max(1, round(0.10 × reference words)).
- A prompt is eligible when it passes at **NFE 20 and 32**. NFE 4 is always included and its WER reported, but it never gates selection, so low-NFE degradation stays visible. 32 of 36 short/medium prompts are eligible; no category/bucket has a gap.
- Rejected prompts: p015 (drops "in total"), p037 (adds "that is" at NFE 20), p022 (Whisper mishears "Valladolid"), p011 (normalizer formats "6:15" as "6.15", likely a false rejection).
- Demo set: one demo-safe prompt per category, then medium prompts from new categories, up to 8.

## Notes for the team

- **Nishant / Tony (#4, #5):** use x-vector conditioning (pass `prompt_audio_path`, not `prompt_text`) and `max_generate_length` ≥ 500. `nfe_benchmark.py` now defaults to 500 and warns when an x-vector output hits the cap. Latency at 500 is not directly comparable to the archived numbers.
- **Arush (#9):** these scores use `openai/whisper-large-v3`. If you use a different model, rerun with `--asr-model <id>`; transcripts are cached per model in `asr_scores.csv`.
- **Sai / Rababb (#1, #2):** pick demo prompts from `safe_demo_prompts.json`. Long prompts are not covered by the archived data.
- **Before rehearsal:** listen to every file in `demo_set/` and tick the checklist. Two NFE-4 contrast files fail the signal check (p010 ends abruptly, p028 has 0.58 s of leading silence).

## Regenerate

From the repository root, with the `dots_tts` environment:

```bash
python experiments/clean_samples/flag_wav_runs.py
python experiments/clean_samples/select_clean_samples.py            # ~2 min on Apple MPS
python experiments/plot_nfe_sweep_comparison.py                     # refreshes report warnings
```
