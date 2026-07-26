#!/bin/bash
# One node's worth of the GPT-2 seed study: one single-GPU training run per
# GPU, all in parallel. Args: specs of the form gpu:kind:seed.
# Invoked on each allocated node by gpt2_seed_study.pbs (via ssh).
set -u
REPO=/path/to/project/omnilens
CKPT_BASE=$REPO/src/checkpoints
cd $REPO
mkdir -p logs

module use /path/to/modulefiles
module load conda
conda activate omnilens
source ~/.hf_env
export HF_HOME=/path/to/hf_cache
export PYTHONPATH=$REPO/src:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

COMMON="--model_name gpt2 --data_source pile --activation_site_preset residual \
 --max_seq_len 1024 --batch_size 32 --tokens_per_step 262144 --num_steps 1000 \
 --warmup_steps 0 --lr 1e-3 --amp --amp_dtype bf16 --save_every 125 --log_every 20 \
 --base_output_dir $CKPT_BASE"
LORA="--lens_type lora --lora_rank 64 --lora_init default_lora"

outdir() {  # kind seed -> canonical auto-derived output dir
  case $1 in
    tuned_kl) echo "$CKPT_BASE/gpt2/tuned/kl/residual/seed$2" ;;
    lora_kl)  echo "$CKPT_BASE/gpt2/lora-r64/kl/residual/init-default/seed$2" ;;
    topk1024) echo "$CKPT_BASE/gpt2/lora-r64/subset_kl-topk-k1024/residual/init-default/seed$2" ;;
    is512)    echo "$CKPT_BASE/gpt2/lora-r64/subset_kl-is-k512-tail512/residual/init-default/seed$2" ;;
    rs512)    echo "$CKPT_BASE/gpt2/lora-r64/subset_kl-is-k0-tail512/residual/init-default/seed$2" ;;
    rs1024)   echo "$CKPT_BASE/gpt2/lora-r64/subset_kl-is-k0-tail1024/residual/init-default/seed$2" ;;
    rs1536)   echo "$CKPT_BASE/gpt2/lora-r64/subset_kl-is-k0-tail1536/residual/init-default/seed$2" ;;
  esac
}

lossargs() {
  case $1 in
    tuned_kl) echo "--lens_type tuned --loss_type kl --kl_chunk_size 128" ;;
    lora_kl)  echo "$LORA --loss_type kl --kl_chunk_size 128" ;;
    topk1024) echo "$LORA --loss_type subset_kl --subset_kl_mode topk --subset_kl_k 1024" ;;
    is512)    echo "$LORA --loss_type subset_kl --subset_kl_mode is --subset_kl_k 512 --subset_kl_k_tail 512" ;;
    rs512)    echo "$LORA --loss_type subset_kl --subset_kl_mode is --subset_kl_k 0 --subset_kl_k_tail 512" ;;
    rs1024)   echo "$LORA --loss_type subset_kl --subset_kl_mode is --subset_kl_k 0 --subset_kl_k_tail 1024" ;;
    rs1536)   echo "$LORA --loss_type subset_kl --subset_kl_mode is --subset_kl_k 0 --subset_kl_k_tail 1536" ;;
  esac
}

run_one() {  # gpu kind seed
  local gpu=$1 kind=$2 seed=$3
  local out; out=$(outdir "$kind" "$seed")
  if [ -f "$out/lens_step_1000.pt" ]; then
    echo "[$(hostname -s)/gpu$gpu] $kind seed$seed already complete, skipping"
    return 0
  fi
  local resume=""
  local last; last=$(ls -v "$out"/lens_step_*.pt 2>/dev/null | tail -1)
  [ -n "$last" ] && resume="--resume_checkpoint $last"
  echo "[$(hostname -s)/gpu$gpu] $(date '+%m-%d %H:%M') starting $kind seed$seed ${resume:+(resume from ${last##*/})}"
  CUDA_VISIBLE_DEVICES=$gpu python -m omnilens.cli.main train \
    $COMMON $(lossargs "$kind") --seed "$seed" $resume \
    >> "$REPO/logs/gpt2seeds_${kind}_s${seed}.log" 2>&1
  echo "[$(hostname -s)/gpu$gpu] $(date '+%m-%d %H:%M') finished $kind seed$seed rc=$?"
}

for spec in "$@"; do
  IFS=: read -r gpu kind seed <<< "$spec"
  run_one "$gpu" "$kind" "$seed" &
done
wait
