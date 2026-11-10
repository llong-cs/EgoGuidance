#!/usr/bin/env bash
# Install LIBERO and the evaluation-client dependencies into the active Python environment.
# Use a separate environment from starVLA, e.g.
#
#   conda create -n libero python=3.10 -y && conda activate libero
#   bash examples/LIBERO/eval_files/install_libero.sh
#
# Optional variables:
#   LIBERO_HOME (playground/LIBERO)                     where LIBERO is cloned
#   LIBERO_CONFIG_PATH (playground/libero_config/libero) where LIBERO's config.yaml is written
#   PYTHON (python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/playground/LIBERO}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/playground/libero_config/libero}"
PYTHON="${PYTHON:-python}"

echo "=== 1/4 Clone LIBERO -> ${LIBERO_HOME}"
if [[ ! -d "${LIBERO_HOME}/libero/libero" ]]; then
  mkdir -p "$(dirname "${LIBERO_HOME}")"
  git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_HOME}"
fi
LIBERO_HOME="$(cd "${LIBERO_HOME}" && pwd)"

echo "=== 2/4 Install Python packages"
# LIBERO loads initial states with torch.load; a CPU build is enough for the simulator client.
"${PYTHON}" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
"${PYTHON}" -m pip install -r "${LIBERO_HOME}/requirements.txt"
"${PYTHON}" -m pip install -e "${LIBERO_HOME}"
# Client dependencies (deployment/model_server/tools/websocket_policy_client.py, video writing).
"${PYTHON}" -m pip install numpy==1.22.4 mujoco==3.2.3 "websockets>=13" msgpack "typing_extensions>=4.4" \
  imageio imageio-ffmpeg

echo "=== 3/4 Write ${LIBERO_CONFIG_PATH}/config.yaml"
# Without this file, importing libero asks for paths interactively.
pkg="${LIBERO_HOME}/libero/libero"
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${pkg}
bddl_files: ${pkg}/bddl_files
init_states: ${pkg}/init_files
datasets: ${pkg}/../datasets
assets: ${pkg}/assets
EOF

echo "=== 4/4 Check"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" "${PYTHON}" - <<'EOF'
from libero.libero import benchmark

suites = benchmark.get_benchmark_dict()
counts = {name: suites[name]().n_tasks for name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]}
print("task counts:", counts)
assert all(n == 10 for n in counts.values()), "expected 10 tasks per suite (is LIBERO-Plus installed instead?)"
EOF
echo "Done. Evaluation scripts use LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
