#!/bin/bash
set -euo pipefail

slot="${PMI_RANK:-${PALS_RANKID:-${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-}}}}"
if [[ -z "${slot}" ]]; then
  echo "[mpi-worker] unable to determine MPI rank" >&2
  exit 2
fi

read -r -a rank_array <<< "${RANKS:?missing RANKS}"
if (( slot < 0 || slot >= ${#rank_array[@]} )); then
  echo "[mpi-worker] slot ${slot} outside rank array of length ${#rank_array[@]}" >&2
  exit 2
fi

rank="${rank_array[$slot]}"
out_dir="${JOB_OUT_DIR:?missing JOB_OUT_DIR}/rank_${rank}"
mkdir -p "${out_dir}"

echo "[mpi-worker] slot=${slot} host=$(hostname) rank=${rank} out_dir=${out_dir}"

/path/to/project/eval_harness/run_pearson_kendall_rank_worker.sh \
  "${rank}" \
  "${out_dir}" 2>&1 | tee "${out_dir}/rank_${rank}.log"
