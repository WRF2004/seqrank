#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:?Set CONFIG to the selected validation configuration}"
# Freeze only after selecting configuration/seeds from validation results.
python -m seqrank.cli freeze-evaluation --config "$CONFIG"
for stage in retrieval ranking e2e; do
  python -m seqrank.cli "evaluate-$stage" --config "$CONFIG" --split test
done
