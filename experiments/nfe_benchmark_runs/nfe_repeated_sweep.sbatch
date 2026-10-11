#!/bin/bash
#SBATCH --job-name=nfe-repeat
#SBATCH --partition=compute
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=shard:rtx_4090:16000
#SBATCH --time=02:00:00
#SBATCH --output=experiments/nfe_benchmark_runs/logs/%j-nfe-repeat.out
set -euo pipefail
cd "$HOME/watai_exp"
source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate dots_tts
python experiments/nfe_benchmark.py \
  --nfe-values 4 6 8 10 12 16 20 24 32 \
  --prompt-limit 1 \
  --repeats 5 \
  --warmup-runs-per-case 1 \
  --precision bfloat16 \
  --max-generate-length 80 \
  --seed 42 \
  --output-dir experiments/nfe_benchmark_runs/nfe_sweep_repeated_20261012
