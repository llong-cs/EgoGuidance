#!/usr/bin/env bash
# Evaluate a checkpoint on LIBERO with one simulator process (run in the LIBERO environment, with
# run_policy_server.sh already serving the same checkpoint).
#
#   CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
#     bash examples/LIBERO/eval_files/eval_libero.sh
#
# Results go to OUT_DIR (default: <run_dir>/eval/libero/<checkpoint name>); re-running with the same
# OUT_DIR resumes. Optional variables:
#   OUT_DIR, HOST (127.0.0.1), PORT (10093), SUITES ("libero_spatial libero_object libero_goal libero_10"),
#   NUM_TRIALS (50), VIDEOS_PER_TASK (1), UNNORM_KEY, LIBERO_PYTHON (python),
#   LIBERO_CONFIG_PATH (playground/libero_config/libero), LIBERO_HOME (added to PYTHONPATH if set),
#   MUJOCO_GL (egl; osmesa for CPU rendering), EVAL_ARGS (extra eval_libero.py flags,
#   e.g. "--max-tasks-per-suite 1" for a smoke test).
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-10093}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
NUM_TRIALS="${NUM_TRIALS:-50}"
VIDEOS_PER_TASK="${VIDEOS_PER_TASK:-1}"
UNNORM_KEY="${UNNORM_KEY:-}"
LIBERO_PYTHON="${LIBERO_PYTHON:-python}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/playground/libero_config/libero}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
EVAL_ARGS="${EVAL_ARGS:-}"

cd "${REPO_ROOT}"
if [[ -z "${OUT_DIR:-}" ]]; then
  if [[ "${CKPT:-}" != */checkpoints/* ]]; then
    echo "Set OUT_DIR, or CKPT=<run_dir>/checkpoints/<name>.pt to place results under <run_dir>/eval/libero/." >&2
    exit 1
  fi
  tag="${CKPT#*/checkpoints/}"
  tag="${tag%.*}"
  OUT_DIR="${CKPT%%/checkpoints/*}/eval/libero/${tag//\//_}"
fi
if [[ ! -f "${LIBERO_CONFIG_PATH}/config.yaml" ]]; then
  echo "Missing ${LIBERO_CONFIG_PATH}/config.yaml; run examples/LIBERO/eval_files/install_libero.sh" \
    "or set LIBERO_CONFIG_PATH." >&2
  exit 1
fi
mkdir -p "${OUT_DIR}"

export LIBERO_CONFIG_PATH MUJOCO_GL
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-${MUJOCO_GL}}"
export PYTHONPATH="${REPO_ROOT}${LIBERO_HOME:+:${LIBERO_HOME}}${PYTHONPATH:+:${PYTHONPATH}}"

ARGS=(
  --benchmark libero
  --host "${HOST}" --port "${PORT}"
  --suites ${SUITES}
  --num-trials-per-task "${NUM_TRIALS}"
  --videos-per-task "${VIDEOS_PER_TASK}"
  --out-dir "${OUT_DIR}"
)
if [[ -n "${UNNORM_KEY}" ]]; then
  ARGS+=(--unnorm-key "${UNNORM_KEY}")
fi

"${LIBERO_PYTHON}" examples/LIBERO/eval_files/eval_libero.py "${ARGS[@]}" ${EVAL_ARGS} 2>&1 | tee -a "${OUT_DIR}/eval.log"
status=${PIPESTATUS[0]}

"${LIBERO_PYTHON}" examples/LIBERO/eval_files/summarize_results.py "${OUT_DIR}"
exit "${status}"
