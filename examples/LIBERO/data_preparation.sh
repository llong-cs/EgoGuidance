#!/usr/bin/env bash
# Download the LIBERO fine-tuning data (LeRobot v2.0 format, four suites, no-op actions removed) and
# register it under playground/Datasets/LEROBOT_LIBERO_DATA (the data_root_dir used by
# configs/finetune/libero.yaml, mixture "libero_all").
#
#   bash examples/LIBERO/data_preparation.sh              # download into playground/Datasets/LEROBOT_LIBERO_DATA
#   bash examples/LIBERO/data_preparation.sh /data/libero # download elsewhere and symlink it there
#
# Requires the Hugging Face CLI (`hf`, from huggingface_hub>=0.34).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LINK="${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA"
DEST="${1:-${DEST:-${LINK}}}"

if ! command -v hf >/dev/null 2>&1; then
  echo "The 'hf' command was not found; install it with: pip install -U \"huggingface_hub>=0.34\"" >&2
  exit 1
fi

mkdir -p "${DEST}"
DEST="$(cd "${DEST}" && pwd)"
for repo in \
  IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot \
  IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot; do
  name="${repo##*/}"
  hf download "${repo}" --repo-type dataset --local-dir "${DEST}/${name}"
  # The gr00t-style loader reads the camera/state/action layout from meta/modality.json.
  cp "${REPO_ROOT}/examples/LIBERO/train_files/modality.json" "${DEST}/${name}/meta/modality.json"
done

if [[ "${DEST}" != "${LINK}" ]]; then
  mkdir -p "$(dirname "${LINK}")"
  if [[ -e "${LINK}" && ! -L "${LINK}" ]]; then
    echo "${LINK} exists and is not a symlink; leaving it unchanged." >&2
  else
    ln -sfn "${DEST}" "${LINK}"
  fi
fi
echo "LIBERO data ready: ${LINK} -> ${DEST}"
