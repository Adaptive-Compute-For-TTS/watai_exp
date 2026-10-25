# Demo-safe samples

Archived RTX 4090 x-vector outputs that are safe to play for an audience: no word errors at NFE 20 under `openai/whisper-large-v3`, ended before the length cap, and passed the signal checks (no clipping, no long silences, not cut off) at NFE 20, 32. NFE 4 is included as a low-step contrast and may contain audible errors; its WER is reported.

Fill in the **Listened** column by ear before rehearsal (Tue Oct 27). Replace a prompt that sounds wrong rather than editing the WAV.

| File | Prompt | Category | NFE | Duration (s) | Latency (s) | RTF | Word errors | WER | Listened |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| [output_p001_nfe4.wav](output_p001_nfe4.wav) | The kettle is already on the stove. | plain | 4 | 2.72 | 1.91 | 0.70 | 0 | 0.00 | ☐ |
| [output_p001_nfe20.wav](output_p001_nfe20.wav) | The kettle is already on the stove. | plain | 20 | 3.04 | 7.70 | 2.53 | 0 | 0.00 | ☐ |
| [output_p001_nfe32.wav](output_p001_nfe32.wav) | The kettle is already on the stove. | plain | 32 | 3.04 | 12.07 | 3.97 | 0 | 0.00 | ☐ |
| [output_p010_nfe4.wav](output_p010_nfe4.wav) | Your total comes to $42.75. | numbers | 4 | 3.52 | 2.54 | 0.72 | 0 | 0.00 | ☐ |
| [output_p010_nfe20.wav](output_p010_nfe20.wav) | Your total comes to $42.75. | numbers | 20 | 3.68 | 9.15 | 2.49 | 0 | 0.00 | ☐ |
| [output_p010_nfe32.wav](output_p010_nfe32.wav) | Your total comes to $42.75. | numbers | 32 | 4.32 | 16.48 | 3.82 | 0 | 0.00 | ☐ |
| [output_p021_nfe4.wav](output_p021_nfe4.wav) | Professor Nguyen studies archaeopteryx fossils. | names_rare_words | 4 | 3.36 | 2.69 | 0.80 | 0 | 0.00 | ☐ |
| [output_p021_nfe20.wav](output_p021_nfe20.wav) | Professor Nguyen studies archaeopteryx fossils. | names_rare_words | 20 | 3.36 | 8.83 | 2.63 | 0 | 0.00 | ☐ |
| [output_p021_nfe32.wav](output_p021_nfe32.wav) | Professor Nguyen studies archaeopteryx fossils. | names_rare_words | 32 | 3.20 | 12.73 | 3.98 | 0 | 0.00 | ☐ |
| [output_p028_nfe4.wav](output_p028_nfe4.wav) | The CEO joined the call at noon. | acronyms_abbreviations | 4 | 2.56 | 1.89 | 0.74 | 0 | 0.00 | ☐ |
| [output_p028_nfe20.wav](output_p028_nfe20.wav) | The CEO joined the call at noon. | acronyms_abbreviations | 20 | 2.40 | 6.38 | 2.66 | 0 | 0.00 | ☐ |
| [output_p028_nfe32.wav](output_p028_nfe32.wav) | The CEO joined the call at noon. | acronyms_abbreviations | 32 | 2.40 | 10.01 | 4.17 | 0 | 0.00 | ☐ |
| [output_p038_nfe4.wav](output_p038_nfe4.wav) | Wow, that was incredible! | prosody | 4 | 2.08 | 2.05 | 0.98 | 0 | 0.00 | ☐ |
| [output_p038_nfe20.wav](output_p038_nfe20.wav) | Wow, that was incredible! | prosody | 20 | 2.56 | 6.85 | 2.68 | 0 | 0.00 | ☐ |
| [output_p038_nfe32.wav](output_p038_nfe32.wav) | Wow, that was incredible! | prosody | 32 | 2.72 | 10.93 | 4.02 | 0 | 0.00 | ☐ |
| [output_p046_nfe4.wav](output_p046_nfe4.wav) | Six slick seals slid south. | phonetic | 4 | 2.72 | 1.98 | 0.73 | 0 | 0.00 | ☐ |
| [output_p046_nfe20.wav](output_p046_nfe20.wav) | Six slick seals slid south. | phonetic | 20 | 2.72 | 7.38 | 2.71 | 0 | 0.00 | ☐ |
| [output_p046_nfe32.wav](output_p046_nfe32.wav) | Six slick seals slid south. | phonetic | 32 | 2.72 | 11.84 | 4.35 | 0 | 0.00 | ☐ |
| [output_p004_nfe4.wav](output_p004_nfe4.wav) | After the meeting ended, she packed her notes into a canvas bag and caught the last bus downtown. | plain | 4 | 7.52 | 5.51 | 0.73 | 0 | 0.00 | ☐ |
| [output_p004_nfe20.wav](output_p004_nfe20.wav) | After the meeting ended, she packed her notes into a canvas bag and caught the last bus downtown. | plain | 20 | 6.08 | 15.37 | 2.53 | 0 | 0.00 | ☐ |
| [output_p004_nfe32.wav](output_p004_nfe32.wav) | After the meeting ended, she packed her notes into a canvas bag and caught the last bus downtown. | plain | 32 | 6.08 | 24.60 | 4.05 | 0 | 0.00 | ☐ |
| [output_p013_nfe4.wav](output_p013_nfe4.wav) | On March 3rd, 2024, the temperature dropped to minus 18 degrees, and the school closed for 2 days. | numbers | 4 | 7.04 | 4.86 | 0.69 | 0 | 0.00 | ☐ |
| [output_p013_nfe20.wav](output_p013_nfe20.wav) | On March 3rd, 2024, the temperature dropped to minus 18 degrees, and the school closed for 2 days. | numbers | 20 | 6.88 | 17.18 | 2.50 | 0 | 0.00 | ☐ |
| [output_p013_nfe32.wav](output_p013_nfe32.wav) | On March 3rd, 2024, the temperature dropped to minus 18 degrees, and the school closed for 2 days. | numbers | 32 | 6.88 | 26.42 | 3.84 | 0 | 0.00 | ☐ |
