#!/usr/bin/env bash
set -euo pipefail
export DATA_ROOT="${DATA_ROOT:-data/kuairand}"
export RAW_ROOT="${RAW_ROOT:-data/raw/KuaiRand-Pure}"
export RUN_ROOT="${RUN_ROOT:-runs/research}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0}"
CONFIG="${CONFIG:-configs/datasets/kuairand_debug.yaml}"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
[[ -d "$RAW_ROOT" ]] || { echo "Missing downloaded/extracted dataset: $RAW_ROOT" >&2; exit 1; }
python -m seqrank.cli schema --config "$CONFIG"
python -m seqrank.cli convert --config "$CONFIG"
python -m seqrank.cli prepare-data --config "$CONFIG"
python -m seqrank.cli verify-data --config "$CONFIG"
python -m seqrank.cli quality --config "$CONFIG"
for stage in retrieval ranking; do
  python -m seqrank.cli "train-$stage" --config "$CONFIG"
  python -m seqrank.cli "evaluate-$stage" --config "$CONFIG" --split valid
done
python -m seqrank.cli evaluate-e2e --config "$CONFIG" --split valid
