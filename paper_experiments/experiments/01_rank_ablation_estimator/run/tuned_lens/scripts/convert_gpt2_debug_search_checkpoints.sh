#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/conda/envs/omnilens/bin/python}"
SRC_ROOT="${SRC_ROOT:-/path/to/data/omnilens/src/checkpoints/gpt2_debug_search}"
OUT_ROOT="${OUT_ROOT:-/path/to/data/tuned-lens/my_lenses/gpt2_debug_search}"

export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

"${PYTHON_BIN}" "${SCRIPT_DIR}/omnieval_harness.py" "${SRC_ROOT}" "${OUT_ROOT}"

printf '\nConverted checkpoints written under:\n%s\n' "${OUT_ROOT}"
find "${OUT_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort
