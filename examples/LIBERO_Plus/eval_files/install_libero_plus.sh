#!/usr/bin/env bash
# Install LIBERO-Plus, its assets and the evaluation-client dependencies into the active Python
# environment. LIBERO-Plus is a drop-in fork of the `libero` package, so use an environment that
# does not contain standard LIBERO, e.g.
#
#   sudo apt-get install -y libmagickwand-dev libexpat1 libfontconfig1-dev   # ImageMagick for `wand`
#   conda create -n libero_plus python=3.10 -y && conda activate libero_plus
#   bash examples/LIBERO_Plus/eval_files/install_libero_plus.sh
#
# Optional variables:
#   LIBERO_PLUS_HOME (playground/LIBERO-plus)                      where LIBERO-Plus is cloned
#   LIBERO_HOME (playground/LIBERO)                                standard LIBERO checkout; only its assets
#                                                                  are used (cloned if missing)
#   ASSETS_ZIP (playground/Datasets/LIBERO-plus/assets.zip)        downloaded from the Hugging Face Hub if missing
#   LIBERO_CONFIG_PATH (playground/libero_config/libero_plus)      where LIBERO-Plus's config.yaml is written
#   PYTHON (python)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
LIBERO_PLUS_HOME="${LIBERO_PLUS_HOME:-${REPO_ROOT}/playground/LIBERO-plus}"
LIBERO_HOME="${LIBERO_HOME:-${REPO_ROOT}/playground/LIBERO}"
ASSETS_ZIP="${ASSETS_ZIP:-${REPO_ROOT}/playground/Datasets/LIBERO-plus/assets.zip}"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-${REPO_ROOT}/playground/libero_config/libero_plus}"
PYTHON="${PYTHON:-python}"

echo "=== 1/5 Clone LIBERO-Plus -> ${LIBERO_PLUS_HOME}"
if [[ ! -d "${LIBERO_PLUS_HOME}/libero/libero" ]]; then
  mkdir -p "$(dirname "${LIBERO_PLUS_HOME}")"
  git clone https://github.com/sylvestf/LIBERO-plus.git "${LIBERO_PLUS_HOME}"
fi
LIBERO_PLUS_HOME="$(cd "${LIBERO_PLUS_HOME}" && pwd)"
pkg="${LIBERO_PLUS_HOME}/libero/libero"

echo "=== 2/5 Install Python packages"
# LIBERO loads initial states with torch.load; a CPU build is enough for the simulator client.
"${PYTHON}" -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cpu
"${PYTHON}" -m pip install -r "${LIBERO_PLUS_HOME}/requirements.txt"
"${PYTHON}" -m pip install -e "${LIBERO_PLUS_HOME}"
# LIBERO-Plus extras (wand, scikit-image) and the client dependencies, resolved together with numpy pinned to
# LIBERO's version: a newer scikit-image/scipy would otherwise upgrade numpy and break robosuite / opencv.
"${PYTHON}" -m pip install numpy==1.22.4 wand==0.6.13 scikit-image==0.21.0 mujoco==3.2.3 "websockets>=13" \
  msgpack "typing_extensions>=4.4" imageio imageio-ffmpeg huggingface_hub

echo "=== 3/5 Assets -> ${pkg}/assets"
# The LIBERO-Plus repository ships no assets. Its assets.zip adds the new objects and perturbed scenes but
# not all of the standard LIBERO assets (e.g. turbosquid_objects/, stable_scanned_objects/, textures/,
# wall.xml), so copy the standard LIBERO assets first and extract assets.zip over them.
if [[ ! -f "${pkg}/assets/.libero_base_copied" ]]; then
  if [[ ! -d "${LIBERO_HOME}/libero/libero/assets" ]]; then
    mkdir -p "$(dirname "${LIBERO_HOME}")"
    git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_HOME}"
  fi
  mkdir -p "${pkg}/assets"
  cp -a "${LIBERO_HOME}/libero/libero/assets/." "${pkg}/assets/"
  touch "${pkg}/assets/.libero_base_copied"
fi
if [[ ! -f "${ASSETS_ZIP}" ]]; then
  mkdir -p "$(dirname "${ASSETS_ZIP}")"
  "${PYTHON}" - "${ASSETS_ZIP}" <<'EOF'
import os
import sys

from huggingface_hub import hf_hub_download

target = os.path.abspath(sys.argv[1])
path = hf_hub_download("Sylvest/LIBERO-plus", "assets.zip", repo_type="dataset", local_dir=os.path.dirname(target))
if os.path.abspath(path) != target:
    os.replace(path, target)
EOF
fi
# Extract every entry below the archive's top-level "assets/" directory (wherever that sits in the archive)
# into <pkg>/assets/, overwriting the standard LIBERO files. Files that already exist with the same size are
# skipped, so an interrupted extraction can be resumed.
"${PYTHON}" - "${ASSETS_ZIP}" "${pkg}/assets" <<'EOF'
import os
import shutil
import sys
import zipfile

zip_path, dest = sys.argv[1], sys.argv[2]
written = skipped = 0
with zipfile.ZipFile(zip_path) as zf:
    for info in zf.infolist():
        parts = info.filename.split("/")
        if info.is_dir() or parts[0] == "__MACOSX" or "assets" not in parts[:-1]:
            continue
        rel = parts[parts.index("assets") + 1 :]
        if any(part in ("", ".", "..") for part in rel):
            continue
        out = os.path.join(dest, *rel)
        if os.path.isfile(out) and os.path.getsize(out) == info.file_size:
            skipped += 1
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with zf.open(info) as src, open(out, "wb") as dst:
            shutil.copyfileobj(src, dst, 1 << 20)
        written += 1
print(f"assets: {written} files written, {skipped} already present")
EOF
for item in articulated_objects new_objects scenes stable_hope_objects stable_scanned_objects textures \
  turbosquid_objects serving_region.xml wall_frames.stl wall.xml; do
  [[ -e "${pkg}/assets/${item}" ]] || { echo "Missing asset: ${pkg}/assets/${item}" >&2; exit 1; }
done

echo "=== 4/5 Write ${LIBERO_CONFIG_PATH}/config.yaml"
# Kept separate from standard LIBERO's config: both packages are named `libero`.
mkdir -p "${LIBERO_CONFIG_PATH}"
cat > "${LIBERO_CONFIG_PATH}/config.yaml" <<EOF
benchmark_root: ${pkg}
bddl_files: ${pkg}/bddl_files
init_states: ${pkg}/init_files
datasets: ${pkg}/../datasets
assets: ${pkg}/assets
EOF

echo "=== 5/5 Check"
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" "${PYTHON}" - "${pkg}" <<'EOF'
import os
import sys

from wand.api import library  # fails here if ImageMagick's MagickWand library is missing

assert hasattr(library, "MagickMotionBlurImage")  # used by LIBERO-Plus's sensor-noise perturbations
from libero.libero import benchmark

suites = benchmark.get_benchmark_dict()
counts = {name: suites[name]().n_tasks for name in ["libero_spatial", "libero_object", "libero_goal", "libero_10"]}
print("task counts:", counts, "total:", sum(counts.values()))
assert all(n > 10 for n in counts.values()), "standard LIBERO is installed instead of LIBERO-Plus"
assert os.path.isfile(os.path.join(sys.argv[1], "benchmark", "task_classification.json"))
EOF
echo "Done. Evaluation scripts use LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH}"
