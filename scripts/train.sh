#!/usr/bin/env bash
# ==============================================================================
# Launch one training run with `accelerate launch` (DeepSpeed ZeRO-2).
#
# Usage:
#   bash scripts/train.sh <config_yaml> [--dotted.key value ...]
#
# Examples:
#   bash scripts/train.sh configs/pretrain/cotrain_openaoe.yaml
#   bash scripts/train.sh configs/finetune/libero.yaml \
#       --run_id my_libero_run --run_root_dir playground/Checkpoints \
#       --trainer.pretrained_checkpoint <run_dir>/checkpoints/steps_20000_pytorch_model.pt
#   CUDA_VISIBLE_DEVICES=0 bash scripts/train.sh configs/finetune/libero.yaml --trainer.max_train_steps 10
#
# * The training script is the config's top-level `train_entry`
#   (default: starVLA/training/train_starvla.py).
# * Outputs go to <run_root_dir>/<run_id>/. `run_id` comes from the config or `--run_id`;
#   `run_root_dir` from the config, `--run_root_dir`, or defaults to playground/Checkpoints.
# * Every other `--dotted.key value` argument overrides the config.
# * Logs are written to <run_root_dir>/<run_id>/logs/train_node<NODE_RANK>_<timestamp>.log.
#
# Topology (standard environment variables):
#   NPROC_PER_NODE  processes per node (default: all visible GPUs, honouring CUDA_VISIBLE_DEVICES)
#   NNODES          number of nodes (default: 1)
#   NODE_RANK       rank of this node (default: 0)
#   MASTER_ADDR     address of node 0 (required when NNODES > 1; default 127.0.0.1)
#   MASTER_PORT     rendezvous port (required when NNODES > 1; default 29500)
# For multi-node runs, start this script on every node with the same arguments and a shared
# filesystem for <run_root_dir>.
#
# Dataset statistics caches go to $STARVLA_STATS_CACHE_DIR (default:
# <run_root_dir>/_dataset_cache), so read-only dataset directories work as well.
# ==============================================================================

set -uo pipefail

die() {
  echo "[train.sh] $*" >&2
  exit 2
}

if [ "$#" -lt 1 ] || [[ "$1" == -* ]]; then
  echo "Usage: bash scripts/train.sh <config_yaml> [--dotted.key value ...]" >&2
  exit 2
fi
config_yaml="$1"
shift
# The remaining arguments ("$@") are config overrides, passed through unchanged.

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)" || die "cannot resolve the repository root"
cd "${repo_root}" || die "cannot enter ${repo_root}"
[ -f "${config_yaml}" ] || die "config file not found: ${config_yaml} (paths are relative to ${repo_root})"
export PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"

# ---- Resolve train_entry / run_id / run_root_dir from the config + overrides ----
runtime_metadata="$(python scripts/resolve_runtime_config.py "${config_yaml}" "$@" | tail -n 1)" \
  || die "failed to read train_entry / run_id / run_root_dir from ${config_yaml}"
IFS=$'\t' read -r train_entry run_id run_root_dir <<< "${runtime_metadata}"
[ -n "${train_entry:-}" ] && [ -n "${run_id:-}" ] && [ -n "${run_root_dir:-}" ] \
  || die "unexpected output from scripts/resolve_runtime_config.py: '${runtime_metadata}'"
[ -f "${train_entry}" ] || die "train_entry does not exist: ${train_entry}"

# ---- Topology ----
is_non_negative_int() { [[ "$1" =~ ^[0-9]+$ ]]; }

count_visible_gpus() {
  if [ -n "${CUDA_VISIBLE_DEVICES+set}" ]; then
    awk -F',' '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}"
  elif command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi -L | grep -c '^GPU'
  else
    python -c "import torch; print(torch.cuda.device_count())"
  fi
}

NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-$(count_visible_gpus)}"
is_non_negative_int "${NNODES}" && [ "${NNODES}" -ge 1 ] || die "NNODES must be a positive integer, got '${NNODES}'"
is_non_negative_int "${NODE_RANK}" && [ "${NODE_RANK}" -lt "${NNODES}" ] \
  || die "NODE_RANK must be in [0, NNODES), got '${NODE_RANK}'"
is_non_negative_int "${NPROC_PER_NODE}" && [ "${NPROC_PER_NODE}" -ge 1 ] \
  || die "no GPU found; set NPROC_PER_NODE or CUDA_VISIBLE_DEVICES (got '${NPROC_PER_NODE}')"
num_processes=$((NNODES * NPROC_PER_NODE))

if [ "${NNODES}" -gt 1 ]; then
  [ -n "${MASTER_ADDR:-}" ] || die "MASTER_ADDR must be set for multi-node training"
  [ -n "${MASTER_PORT:-}" ] || die "MASTER_PORT must be set for multi-node training"
else
  MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  MASTER_PORT="${MASTER_PORT:-29500}"
  # A single host needs no network transport between GPUs; bootstrap over loopback.
  export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
fi
export MASTER_ADDR MASTER_PORT
if [ "${num_processes}" -eq 1 ]; then
  # Make the single process a well-formed distributed job of size 1 for DeepSpeed.
  export RANK="${RANK:-0}" LOCAL_RANK="${LOCAL_RANK:-0}" WORLD_SIZE="${WORLD_SIZE:-1}"
fi

# Surface NCCL failures as errors on every rank instead of a silent hang.
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export TOKENIZERS_PARALLELISM=false
export STARVLA_STATS_CACHE_DIR="${STARVLA_STATS_CACHE_DIR:-${run_root_dir}/_dataset_cache}"

# ---- Logging ----
log_dir="${run_root_dir}/${run_id}/logs"
mkdir -p "${log_dir}" "${STARVLA_STATS_CACHE_DIR}" || die "cannot create ${log_dir}"
log_file="${log_dir}/train_node${NODE_RANK}_$(date +%Y%m%d_%H%M%S).log"

{
  echo "[$(date '+%F %T')] START run_id=${run_id}"
  echo "  config     : ${config_yaml}"
  echo "  entry      : ${train_entry}"
  echo "  output dir : ${run_root_dir}/${run_id}"
  echo "  overrides  : $*"
  echo "  processes  : ${num_processes} (NNODES=${NNODES} x NPROC_PER_NODE=${NPROC_PER_NODE}, NODE_RANK=${NODE_RANK})"
  echo "  rendezvous : ${MASTER_ADDR}:${MASTER_PORT}"
  echo "  log        : ${log_file}"
} | tee "${log_file}"

# ---- Launch ----
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes "${num_processes}" \
  --num_machines "${NNODES}" \
  --machine_rank "${NODE_RANK}" \
  --main_process_ip "${MASTER_ADDR}" \
  --main_process_port "${MASTER_PORT}" \
  --rdzv_backend static \
  "${train_entry}" \
  --config_yaml "${config_yaml}" \
  "$@" \
  --run_id "${run_id}" \
  --run_root_dir "${run_root_dir}" \
  2>&1 | tee -a "${log_file}"
rc="${PIPESTATUS[0]}"

echo "[$(date '+%F %T')] DONE run_id=${run_id} exit_code=${rc}" | tee -a "${log_file}"
exit "${rc}"
