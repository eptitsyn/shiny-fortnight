#!/usr/bin/env bash
# Extract cached features (hidden states + next-token metrics) on 2 GPUs.
# Usage:
#   ./scripts/extract.sh [hydra overrides...]
#   CUDA_VISIBLE_DEVICES=2,3 ./scripts/extract.sh extract.splits='[validation,test]'

set -euo pipefail

cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export TOKENIZERS_PARALLELISM=false

torchrun \
    --standalone \
    --nnodes=1 \
    --nproc_per_node=2 \
    extract_features.py "$@"
