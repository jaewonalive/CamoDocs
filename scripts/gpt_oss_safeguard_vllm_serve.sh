#!/bin/bash
# Launch vLLM-serve hosting openai/gpt-oss-safeguard-20b for use as the
# LLM Filter critic. This is required before running any eval script
# with --removal_method llm_critic (e.g. scripts/llm_filter_eval.sh).
#
# *** TWO-TERMINAL WORKFLOW ***
#
# Terminal 1 (this script):
#   bash scripts/gpt_oss_safeguard_vllm_serve.sh
# Wait for "Application startup complete." then check:
#   curl -s http://localhost:8001/v1/models
#
# Terminal 2 (eval runs, repeatable):
#   bash scripts/llm_filter_eval.sh
# vLLM keeps serving the critic between each run.
#
# When done, manually kill:
#   pkill -f 'vllm serve'
#
# *** COEXISTENCE MODE ***
#
# --gpu-memory-utilization 0.7 reserves ~34 GB / 48 GB per A6000 for the
# critic, leaving ~14 GB per GPU for the VICTIM model to coexist on the
# same GPUs. This is the value used for the reported Llama-3.1-8B and
# Qwen3-8B results:
#   Qwen3-8B bf16 TP=4    : ~4 GB / GPU  -> comfortable
#   Llama-3.1-8B bf16     : ~4 GB / GPU  -> comfortable
# A larger victim needs a smaller critic reservation. Mixtral-8x7B bf16
# TP=4 takes ~24 GB / GPU, which does not fit alongside 34 GB, so those
# runs used --gpu-memory-utilization 0.3 (~14 GB critic, 24 + 14 < 48).

set -euo pipefail

# *** IMPORTANT *** This script requires a SEPARATE conda env from the
# main eval pipeline because gpt-oss-safeguard-20b needs a newer vLLM /
# transformers / torch stack. See requirements_vllm.txt. Activate it before
# running:
#   conda activate camodocs_vllm   # whatever you named the env


: "${HF_TOKEN:?HF_TOKEN must be set to download openai/gpt-oss-safeguard-20b weights}"

MODEL=${MODEL:-openai/gpt-oss-safeguard-20b}
PORT=${PORT:-8001}
TP=${TP:-4}                 # must equal the number of visible GPUs

# Exported, not a bare assignment: the latter stays in this shell and would
# never reach the `vllm serve` child process.
#
# Defaults to the first four GPUs, but never overrides a value already set.
# That matters under a scheduler: SLURM sets CUDA_VISIBLE_DEVICES to the GPUs
# it allocated you, and clobbering it would send this job onto GPUs belonging
# to someone else. Override explicitly if you want different devices:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/gpt_oss_safeguard_vllm_serve.sh
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=================================================================="
echo "[$(date '+%F %T')] starting vLLM serve"
echo "  model: $MODEL"
echo "  port:  $PORT"
echo "  TP:    $TP"
echo "  GPUs:  $CUDA_VISIBLE_DEVICES"
echo "=================================================================="

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --tensor-parallel-size $TP \
    --gpu-memory-utilization 0.7 \
    --max-model-len 8192 \
    --port $PORT
