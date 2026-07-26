#!/bin/bash
# Sequential submitter: debug queue allows only one queued job per user.
cd /path/to/project/memory_injections/experiments/tweak_factor_analysis
prev=7243096
wait_done() {
  while qstat "$1" >/dev/null 2>&1; do
    st=$(qstat -f "$1" 2>/dev/null | awk '/job_state/{print $3}')
    [[ "$st" == "F" || -z "$st" ]] && return
    sleep 90
  done
}
for script in run_llama8b_alllens.pbs run_llama70b_alllens_hand.pbs run_llama70b_alllens_2wmh.pbs; do
  wait_done "$prev"
  id=$(qsub "$script") || { echo "$(date) SUBMIT FAILED: $script"; exit 1; }
  echo "$(date) submitted $script -> $id"
  prev=$id
done
wait_done "$prev"
echo "$(date) chain complete (last job $prev finished)"
