#!/bin/bash
#SBATCH --job-name=nfe-54prompt
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=shard:rtx_4090:16000
#SBATCH --time=24:00:00
#SBATCH --array=0-1%1
#SBATCH --output=experiments/nfe_benchmark_runs/logs/%A_%a-nfe-repeat.out
set -euo pipefail
cd "$HOME/watai_exp"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dots_tts
PROMPT_LIMIT=27
PROMPT_OFFSET=$((SLURM_ARRAY_TASK_ID * PROMPT_LIMIT))
python experiments/nfe_benchmark.py \
  --manifest experiments/benchmark_prompts_v1.csv \
  --reference-audio experiments/my_files/abhi1.wav \
  --reference-text "Hey, Adam here. Let's create something that feels real, sounds human, and connects every time." \
  --nfe-values 4 6 8 10 12 16 20 24 32 \
  --prompt-offset "$PROMPT_OFFSET" \
  --prompt-limit "$PROMPT_LIMIT" \
  --repeats 1 \
  --warmup-runs-per-case 1 \
  --precision bfloat16 \
  --max-generate-length 80 \
  --seed 42 \
  --ode-method euler \
  --guidance-scale 1.2 \
  --speaker-scale 1.5 \
  --max-sequence-length 2048 \
  --no-optimize \
  --resume \
  --output-dir "experiments/nfe_benchmark_runs/nfe_54_prompts_v1_part_${SLURM_ARRAY_TASK_ID}"
