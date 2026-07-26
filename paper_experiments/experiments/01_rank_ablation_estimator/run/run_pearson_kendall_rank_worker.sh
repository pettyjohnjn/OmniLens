#!/bin/bash
set -euo pipefail

RANK="${1:?usage: run_pearson_kendall_rank_worker.sh RANK OUT_DIR}"
OUT_DIR="${2:?usage: run_pearson_kendall_rank_worker.sh RANK OUT_DIR}"

cd /path/to/project/eval_harness

PYTHON_BIN="${PYTHON_BIN:-/path/to/conda/envs/tuned_lens_env/bin/python3.9}"

export HF_HOME="/path/to/hf_cache"
export HF_HUB_CACHE="${HF_HOME}/hub"
export TRANSFORMERS_CACHE="${HF_HOME}/transformers"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
export XDG_CACHE_HOME="/path/to/cache"
export PYTHONPATH="/path/to/project/eval_harness:/path/to/project/omnilens/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export RAYON_NUM_THREADS="${RAYON_NUM_THREADS:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-${USER:-omnilens}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "[worker] host=$(hostname)"
echo "[worker] rank=${RANK}"
echo "[worker] out_dir=${OUT_DIR}"
echo "[worker] families=${FAMILIES:-lora_full lora_topk lora_head_is}"
echo "[worker] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[worker] python=${PYTHON_BIN}"

EXTRA_ARGS=()
if [[ -n "${RUN_REGEX:-}" ]]; then
  EXTRA_ARGS+=(--run-regex "${RUN_REGEX}")
fi
if [[ -n "${EXCLUDE_RUN_REGEX:-}" ]]; then
  EXTRA_ARGS+=(--exclude-run-regex "${EXCLUDE_RUN_REGEX}")
fi

read -r -a FAMILIES_ARRAY <<< "${FAMILIES:-lora_full lora_topk lora_head_is}"

"${PYTHON_BIN}" compute_gpt2_pearson_kendall_token_level.py \
  --model-name "${MODEL_NAME:-gpt2}" \
  --tokens "${TOKENS:-131072}" \
  --max-raw-docs "${MAX_RAW_DOCS:-8192}" \
  --positions "${POSITIONS:-8192}" \
  --kendall-k "${KENDALL_K:-100}" \
  --batch-size "${BATCH_SIZE:-1}" \
  --max-seq-len "${MAX_SEQ_LEN:-1024}" \
  --data-name "${DATA_NAME:-test}" \
  --dtype "${DTYPE:-bfloat16}" \
  --out-dir "${OUT_DIR}" \
  --families "${FAMILIES_ARRAY[@]}" \
  --ranks "${RANK}" \
  "${EXTRA_ARGS[@]}"
