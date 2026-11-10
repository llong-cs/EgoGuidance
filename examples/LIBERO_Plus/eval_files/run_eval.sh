#!/usr/bin/env bash
# Single-node, multi-GPU evaluation: starts N policy servers and M simulator workers, waits for the
# workers, stops the servers and summarizes the results. Run it from the starVLA environment (servers use
# STARVLA_PYTHON); the simulator workers use LIBERO_PYTHON from the LIBERO-Plus (or LIBERO) environment.
#
#   LIBERO_PYTHON=/path/to/envs/libero_plus/bin/python GPUS="0 1 2 3 4 5 6 7" \
#   CKPT=playground/Checkpoints/<run_id>/checkpoints/steps_20000_pytorch_model.pt \
#     bash examples/LIBERO_Plus/eval_files/run_eval.sh
#
# Topology: SERVERS_PER_GPU servers per GPU in GPUS (N = #GPUS * SERVERS_PER_GPU, ports BASE_PORT..);
# WORKERS_PER_SERVER workers per server (M = N * WORKERS_PER_SERVER). Worker w connects to server w % N and
# renders (EGL) on that server's GPU. Each worker takes every M-th unit of work; results are written per
# episode, so re-running the same command resumes an interrupted evaluation.
#
# Variables (defaults in parentheses):
#   CKPT (required)            checkpoint .pt; its run directory holds config.yaml and dataset_statistics.json
#   LIBERO_PYTHON (required)   python of the simulator environment
#   BENCHMARK (libero_plus)    libero_plus | libero
#   OUT_DIR                    (<run_dir>/eval/<BENCHMARK>/<checkpoint name>)
#   GPUS ("0")                 space-separated GPU ids
#   SERVERS_PER_GPU (1), WORKERS_PER_SERVER (4), BASE_PORT (10093)
#   SUITES ("libero_spatial libero_object libero_goal libero_10")
#   NUM_TRIALS (1 for libero_plus, 50 for libero)       episodes per task
#   EPISODE_CHUNK (0 for libero_plus, 10 for libero)    episodes per work unit (0 = whole task)
#   VIDEOS_PER_TASK (0 for libero_plus, 1 for libero)
#   UNITS_PER_PROCESS (8)      restart a worker process after this many units (frees renderer memory)
#   MAX_RESTARTS (20)          give up on a worker after this many crashes
#   LIBERO_CONFIG_PATH (playground/libero_config/<BENCHMARK>), LIBERO_HOME (added to PYTHONPATH if set)
#   MUJOCO_GL (egl; osmesa renders on the CPU), STARVLA_PYTHON (python), USE_BF16 (1), UNNORM_KEY
#   SERVER_START_TIMEOUT (900 s), WORKER_START_DELAY (5 s), EVAL_ARGS (extra eval_libero.py flags)
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT="${CKPT:?Set CKPT=<run_dir>/checkpoints/<name>.pt}"
LIBERO_PYTHON="${LIBERO_PYTHON:?Set LIBERO_PYTHON to the python of the LIBERO-Plus (or LIBERO) environment}"
BENCHMARK="${BENCHMARK:-libero_plus}"
case "${BENCHMARK}" in
  libero_plus) _trials=1; _chunk=0; _videos=0 ;;
  libero) _trials=50; _chunk=10; _videos=1 ;;
  *) echo "BENCHMARK must be libero_plus or libero, got ${BENCHMARK}" >&2; exit 2 ;;
esac
NUM_TRIALS="${NUM_TRIALS:-${_trials}}"
EPISODE_CHUNK="${EPISODE_CHUNK:-${_chunk}}"
VIDEOS_PER_TASK="${VIDEOS_PER_TASK:-${_videos}}"
GPUS="${GPUS:-0}"
SERVERS_PER_GPU="${SERVERS_PER_GPU:-1}"
WORKERS_PER_SERVER="${WORKERS_PER_SERVER:-4}"
BASE_PORT="${BASE_PORT:-10093}"
SUITES="${SUITES:-libero_spatial libero_object libero_goal libero_10}"
UNITS_PER_PROCESS="${UNITS_PER_PROCESS:-8}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/playground/libero_config/${BENCHMARK}}"
MUJOCO_GL="${MUJOCO_GL:-egl}"
STARVLA_PYTHON="${STARVLA_PYTHON:-python}"
USE_BF16="${USE_BF16:-1}"
UNNORM_KEY="${UNNORM_KEY:-}"
SERVER_START_TIMEOUT="${SERVER_START_TIMEOUT:-900}"
WORKER_START_DELAY="${WORKER_START_DELAY:-5}"
EVAL_ARGS="${EVAL_ARGS:-}"

# ---- checks and paths --------------------------------------------------------------------------------
cd "${REPO_ROOT}"
[[ -f "${CKPT}" ]] || { echo "Checkpoint not found: ${CKPT}" >&2; exit 1; }
CKPT="$(cd "$(dirname "${CKPT}")" && pwd)/$(basename "${CKPT}")"
if [[ -z "${OUT_DIR:-}" ]]; then
  [[ "${CKPT}" == */checkpoints/* ]] || { echo "Set OUT_DIR (CKPT is not under <run_dir>/checkpoints/)" >&2; exit 1; }
  tag="${CKPT#*/checkpoints/}"
  tag="${tag%.*}"
  OUT_DIR="${CKPT%%/checkpoints/*}/eval/${BENCHMARK}/${tag//\//_}"
fi
[[ -f "${LIBERO_CONFIG_PATH}/config.yaml" ]] \
  || { echo "Missing ${LIBERO_CONFIG_PATH}/config.yaml (run the install script or set LIBERO_CONFIG_PATH)" >&2; exit 1; }
case "${MUJOCO_GL}" in egl|osmesa) ;; *) echo "MUJOCO_GL must be egl or osmesa" >&2; exit 2 ;; esac

read -ra GPU_ARR <<< "${GPUS}"
NGPU=${#GPU_ARR[@]}
NSERVER=$(( NGPU * SERVERS_PER_GPU ))
NWORKER=$(( NSERVER * WORKERS_PER_SERVER ))
(( NSERVER > 0 && NWORKER > 0 )) || { echo "Need at least one GPU, server and worker" >&2; exit 2; }
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# One BLAS/OpenMP thread budget per process; unbounded thread pools oversubscribe the CPU badly.
NCPU=$(nproc 2>/dev/null || echo 8)
THREADS=$(( NCPU / (NSERVER + NWORKER) ))
(( THREADS >= 1 )) || THREADS=1
export OMP_NUM_THREADS="${THREADS}" MKL_NUM_THREADS="${THREADS}" OPENBLAS_NUM_THREADS="${THREADS}"
export PYTHONPATH="${REPO_ROOT}${LIBERO_HOME:+:${LIBERO_HOME}}${PYTHONPATH:+:${PYTHONPATH}}"

echo "benchmark=${BENCHMARK} suites=[${SUITES}] trials/task=${NUM_TRIALS} episode_chunk=${EPISODE_CHUNK}"
echo "checkpoint=${CKPT}"
echo "servers=${NSERVER} (GPUs ${GPUS}) workers=${NWORKER} render=${MUJOCO_GL} threads/process=${THREADS}"
echo "results=${OUT_DIR}"

SERVER_PIDS=()
WORKER_PIDS=()
cleanup() {
  trap - EXIT INT TERM
  local pid
  for pid in ${WORKER_PIDS[@]+"${WORKER_PIDS[@]}"} ${SERVER_PIDS[@]+"${SERVER_PIDS[@]}"}; do
    kill "${pid}" 2>/dev/null
  done
  wait 2>/dev/null
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# ---- 1) policy servers ----------------------------------------------------------------------------------
server_gpu() { echo "${GPU_ARR[$(( $1 % NGPU ))]}"; }
for (( s = 0; s < NSERVER; s++ )); do
  port=$(( BASE_PORT + s ))
  if (exec 3<>"/dev/tcp/127.0.0.1/${port}") 2>/dev/null; then
    echo "Port ${port} is already in use (a server from an earlier run?); set BASE_PORT." >&2
    exit 1
  fi
  args=(deployment/model_server/server_policy.py --ckpt_path "${CKPT}" --port "${port}" --idle_timeout -1)
  [[ "${USE_BF16}" == "1" ]] && args+=(--use_bf16)
  [[ -n "${UNNORM_KEY}" ]] && args+=(--unnorm_key "${UNNORM_KEY}")
  CUDA_VISIBLE_DEVICES="$(server_gpu "${s}")" "${STARVLA_PYTHON}" "${args[@]}" > "${LOG_DIR}/server_${s}.log" 2>&1 &
  SERVER_PIDS+=($!)
done

echo "Waiting for ${NSERVER} server(s) to load the checkpoint ..."
deadline=$(( SECONDS + SERVER_START_TIMEOUT ))
for (( s = 0; s < NSERVER; s++ )); do
  until grep -q "Server running" "${LOG_DIR}/server_${s}.log" 2>/dev/null; do
    if ! kill -0 "${SERVER_PIDS[$s]}" 2>/dev/null || (( SECONDS > deadline )); then
      echo "Server ${s} failed to start; last lines of ${LOG_DIR}/server_${s}.log:" >&2
      tail -n 30 "${LOG_DIR}/server_${s}.log" >&2
      exit 1
    fi
    sleep 2
  done
done
echo "All servers are up."

# ---- 2) simulator workers -------------------------------------------------------------------------------
# A worker process exits with 0 when its units are done, 3 when it asks to be restarted (UNITS_PER_PROCESS),
# and anything else on a crash; crashed workers are restarted (completed episodes are kept) up to MAX_RESTARTS.
run_worker() {
  local w="$1" port="$2" gpu="$3" crashes=0 rc child=""
  local -a render_env
  trap 'kill ${child:+"${child}"} 2>/dev/null; exit 143' TERM INT
  if [[ "${MUJOCO_GL}" == "egl" ]]; then
    render_env=(CUDA_VISIBLE_DEVICES="${gpu}" MUJOCO_EGL_DEVICE_ID="${gpu}" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl)
  else
    render_env=(CUDA_VISIBLE_DEVICES= MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa)
  fi
  while true; do
    env -u MUJOCO_EGL_DEVICE_ID "${render_env[@]}" LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
      "${LIBERO_PYTHON}" examples/LIBERO/eval_files/eval_libero.py \
      --benchmark "${BENCHMARK}" --host 127.0.0.1 --port "${port}" \
      --suites ${SUITES} --num-trials-per-task "${NUM_TRIALS}" --episode-chunk "${EPISODE_CHUNK}" \
      --videos-per-task "${VIDEOS_PER_TASK}" --max-units-per-process "${UNITS_PER_PROCESS}" \
      --worker-index "${w}" --num-workers "${NWORKER}" --out-dir "${OUT_DIR}" \
      ${UNNORM_KEY:+--unnorm-key "${UNNORM_KEY}"} ${EVAL_ARGS} &
    child=$!
    wait "${child}"
    rc=$?
    child=""
    case "${rc}" in
      0) return 0 ;;
      3) ;;
      *)
        crashes=$(( crashes + 1 ))
        if (( crashes > MAX_RESTARTS )); then
          echo "worker ${w}: giving up after ${crashes} failures (last exit code ${rc})"
          return "${rc}"
        fi
        echo "worker ${w}: exit code ${rc}, restarting (${crashes}/${MAX_RESTARTS})"
        sleep 5
        ;;
    esac
  done
}

for (( w = 0; w < NWORKER; w++ )); do
  s=$(( w % NSERVER ))
  run_worker "${w}" "$(( BASE_PORT + s ))" "$(server_gpu "${s}")" >> "${LOG_DIR}/worker_${w}.log" 2>&1 &
  WORKER_PIDS+=($!)
  # Staggered start: many renderer contexts created at the same moment on one GPU can crash the EGL driver.
  sleep "${WORKER_START_DELAY}"
done
echo "Started ${NWORKER} worker(s); logs in ${LOG_DIR}. Progress: python examples/LIBERO/eval_files/summarize_results.py ${OUT_DIR}"

failed=0
for (( w = 0; w < NWORKER; w++ )); do
  wait "${WORKER_PIDS[$w]}" || { echo "worker ${w} failed; see ${LOG_DIR}/worker_${w}.log"; failed=1; }
done
WORKER_PIDS=()
cleanup  # stop the servers

# ---- 3) results -----------------------------------------------------------------------------------------
"${LIBERO_PYTHON}" examples/LIBERO/eval_files/summarize_results.py "${OUT_DIR}"
if (( failed )); then
  echo "Some workers failed; re-run the same command to finish the remaining episodes." >&2
  exit 1
fi
