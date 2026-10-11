# Repeated NFE sweep summary

One warmup and five measured runs per NFE, one prompt, fixed seed 42, bfloat16, Euler solver, RTX 4090.

| NFE | Median latency (s) | RTF | Median audio (s) | Latency IQR (s) | WAV status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 6.96 | 0.76 | 9.12 | 6.94 to 7.10 | 5 files; machine checks passed |
| 6 | 9.18 | 1.01 | 9.12 | 8.95 to 9.41 | 5 files; machine checks passed |
| 8 | 11.51 | 1.26 | 9.12 | 11.15 to 12.78 | 5 files; machine checks passed |
| 10 | 13.78 | 1.51 | 9.12 | 13.06 to 15.48 | 5 files; machine checks passed |
| 12 | 13.77 | 1.69 | 8.16 | 13.72 to 14.23 | 5 files; machine checks passed |
| 16 | 19.45 | 2.21 | 8.80 | 18.70 to 19.48 | 5 files; machine checks passed |
| 20 | 23.73 | 2.65 | 8.96 | 23.28 to 23.91 | 5 files; machine checks passed |
| 24 | 27.43 | 3.06 | 8.96 | 27.32 to 27.85 | 5 files; machine checks passed |
| 32 | 36.09 | 3.96 | 9.12 | 35.92 to 36.23 | 5 files; machine checks passed |

Audio checks: 45 WAVs readable; sample rate(s) 48000; duration range 8.16 to 9.12 s; maximum peak 0.4787; max near-silence fraction (<1e-4) 17.5427%; max clipped samples (>=0.999) 0.00000%.

Informal listening notes are pending. File decoding and signal checks do not assess pronunciation, naturalness, or audible artifacts.
