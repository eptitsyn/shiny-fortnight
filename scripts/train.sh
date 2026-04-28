#!/usr/bin/env bash
# Launch PAWN training on 2 GPUs via torchrun (DDP, HuggingFace Trainer).
# Usage:
#   ./scripts/train.sh [hydra overrides...]
#   CUDA_VISIBLE_DEVICES=2,3 ./scripts/train.sh training.learning_rate=1e-3
#
# Examples:
#   ./scripts/train.sh                                    # defaults
#   ./scripts/train.sh model=pawn_simple                  # swap model preset
#   ./scripts/train.sh data=mage_llama training.bf16=true # llama backbone

set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    train.py "$@"
