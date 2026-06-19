#!/usr/bin/env bash
# set_nnunet_env.sh — Export nnUNet environment variables from configs/paths.yaml
#
# Usage:
#   source scripts/set_nnunet_env.sh
#
# After sourcing, the following variables are set:
#   nnUNet_raw, nnUNet_preprocessed, nnUNet_results

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="${REPO_ROOT}/configs/paths.yaml"

# Use the repo's venv Python if available; otherwise fall back to system Python
if [ -x "${REPO_ROOT}/.venv/bin/python3" ]; then
    _PYTHON="${REPO_ROOT}/.venv/bin/python3"
elif command -v python3 &>/dev/null; then
    _PYTHON="python3"
else
    _PYTHON="python"
fi

_yaml_get() {
    # Usage: _yaml_get <yaml_file> <dot.separated.key>
    ${_PYTHON} - "$1" "$2" <<'EOF'
import sys, yaml, os
cfg_path, key = sys.argv[1], sys.argv[2]
with open(cfg_path) as f:
    cfg = yaml.safe_load(f)
parts = key.split(".")
val = cfg
for p in parts:
    val = val[p]
root = os.environ.get("PANCREAS_CYST_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(cfg_path))))
val = str(val).replace("${PANCREAS_CYST_ROOT}", root).replace("${root}", root)
print(val)
EOF
}

export PANCREAS_CYST_ROOT="${PANCREAS_CYST_ROOT:-${REPO_ROOT}}"
export nnUNet_raw="$(_yaml_get "${CONFIG}" "nnunet.raw")"
export nnUNet_preprocessed="$(_yaml_get "${CONFIG}" "nnunet.preprocessed")"
export nnUNet_results="$(_yaml_get "${CONFIG}" "nnunet.results")"

echo "[set_nnunet_env] PANCREAS_CYST_ROOT = ${PANCREAS_CYST_ROOT}"
echo "[set_nnunet_env] nnUNet_raw         = ${nnUNet_raw}"
echo "[set_nnunet_env] nnUNet_preprocessed = ${nnUNet_preprocessed}"
echo "[set_nnunet_env] nnUNet_results      = ${nnUNet_results}"
