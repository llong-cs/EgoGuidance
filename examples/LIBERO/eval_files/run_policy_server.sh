#!/usr/bin/env bash
# Start the policy server for one checkpoint (run in the starVLA environment, from any directory).
#
#   CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
#     bash examples/LIBERO/eval_files/run_policy_server.sh
#
# The checkpoint's run directory must also contain config.yaml and dataset_statistics.json
# (written by training). Optional variables:
#   GPU_ID (0), PORT (10093), USE_BF16 (1), UNNORM_KEY (auto), IDLE_TIMEOUT (1800 s; -1 = never),
#   STARVLA_PYTHON (python).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT="${CKPT:?Set CKPT, e.g. CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt}"
GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-10093}"
USE_BF16="${USE_BF16:-1}"
UNNORM_KEY="${UNNORM_KEY:-}"
IDLE_TIMEOUT="${IDLE_TIMEOUT:-1800}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"

if [[ ! -f "${CKPT}" ]]; then
  echo "Checkpoint not found: ${CKPT}" >&2
  exit 1
fi
CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

CMD=(
  "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
  --ckpt_path "${CKPT}"
  --port "${PORT}"
  --idle_timeout "${IDLE_TIMEOUT}"
)
if [[ "${USE_BF16}" == "1" ]]; then
  CMD+=(--use_bf16)
fi
if [[ -n "${UNNORM_KEY}" ]]; then
  CMD+=(--unnorm_key "${UNNORM_KEY}")
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" exec "${CMD[@]}"
