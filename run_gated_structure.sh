#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="/data/ranran/my_ptm/多位点"

python3 "${SCRIPT_DIR}/run_mt_esm_prott5_gated_structure.py" \
  --experiment-name mt_esm_prott5_custom \
  --out-root "${ROOT}/experiment" \
  --device cuda \
  --require-cuda \
  "$@"
