#!/usr/bin/env bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES="${GPU_IDS:?Set allocated GPU IDs, e.g. GPU_IDS=2,3}"
NPROC="${NPROC:-2}"
CONFIG="${CONFIG:-configs/ddp.yaml}"
STAGE="${STAGE:-ranking}"
[[ -f "$CONFIG" ]] || { echo "Missing config: $CONFIG" >&2; exit 1; }
[[ "$STAGE" == ranking || "$STAGE" == retrieval ]] || { echo "Invalid STAGE" >&2; exit 1; }
torchrun --standalone --nproc_per_node="$NPROC" -m seqrank.cli "train-$STAGE" --config "$CONFIG" "$@"
