#!/bin/bash
# Example: CamoDocs attack + LLM Filter defense, HotpotQA, Qwen3-8B victim.
#
# vLLM serve (openai/gpt-oss-safeguard-20b) runs on a separate host; this
# script connects to it via LLM_CRITIC_URL. Set LLM_CRITIC_URL to override
# the default http://localhost:8001/v1, e.g.:
#
#   export LLM_CRITIC_URL="http://<host>:8001/v1"
#
# On a single machine, launch vLLM with
# --gpu-memory-utilization 0.7 (see scripts/gpt_oss_safeguard_vllm_serve.sh)
# so the victim and critic can coexist on the same GPUs. evaluate.py's
# vLLM-critic branch leaves the server alive and proceeds; no manual
# pkill / handoff is required.
#
# Pipeline:
#   (a) evaluate.py talks to the vLLM endpoint to filter docs
#       (one per-document call, --removal_method llm_critic).
#   (b) Accumulate filtered top-k contents across queries.
#   (c) The victim model loads and generates answers (vLLM stays alive,
#       so the same critic can serve subsequent eval runs).

set -euo pipefail




: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (LLM judge)}"
: "${HF_TOKEN:?HF_TOKEN must be set for gated models}"
: "${REPO_ROOT:?REPO_ROOT must be set to this repository (absolute path)}"
: "${DATA_ROOT:?DATA_ROOT must be set to where adv-text artifacts live}"

EXP_ID_NUM=${EXP_ID_NUM:-llm_filter_eval-}

EVAL_MODEL_CODE="contriever"
EVAL_DATASET="${EVAL_DATASET:-hotpotqa}"
SPLIT="${SPLIT:-test}"
TOP_K="5"
GPU_ID="0"
SCORE_FUNCTION="dot"

TEST_NO_ATTACK=False
ATTACK_METHOD="LM_targeted"
DEFEND_METHOD="none"
REMOVAL_METHOD="llm_critic"

M="100"
REPEAT_TIMES="10"
SEED="12"
ADV_PER_QUERY=10
NOTE="None"

MODEL_NAME="llama"
MODEL_NAME_FOR_LOG="${MODEL_NAME//\//-}"
LM_DEPLOY_TP=${LM_DEPLOY_TP:-4}   # must equal the number of visible GPUs
LM_DEPLOY_SESSION_LEN=131072

ATTACK_SAMPLE_NUM=1000

LLM_JUDGE=True
LLM_JUDGE_MODEL=gpt-4.1-mini-2025-04-14

LOG_NAME=$EXP_ID_NUM-$EVAL_DATASET-$EVAL_MODEL_CODE-$MODEL_NAME_FOR_LOG-camodocs-llm_critic-M$M-rep$REPEAT_TIMES-$NOTE.log

# Example uses the bundled CamoDocs HotpotQA output.
CUSTOM_ATTACK_PATH=${CUSTOM_ATTACK_PATH:-${REPO_ROOT}/data_examples/camodocs_${EVAL_DATASET}_adv_text_merged.json}
TARGET_QUERIES=${TARGET_QUERIES:-${REPO_ROOT}/target_queries_fixed/${EVAL_DATASET}_target_queries_fixed.json}

QUERY_OMITTED=${QUERY_OMITTED:-True}   # set False to reproduce the PoisonedRAG baseline

# vLLM critic endpoint. Override LLM_CRITIC_URL to point at a remote host.
LLM_CRITIC_URL="${LLM_CRITIC_URL:-http://localhost:8001/v1}"
LLM_CRITIC_MODEL="vllm:${LLM_CRITIC_URL}:openai/gpt-oss-safeguard-20b"

# Fail-fast guard: the LLM-critic endpoint must be up before we start.
# A dead critic causes silent corruption (every doc gets erased as
# fallback, producing misleading low-ASR results).
LLM_CRITIC_HEALTH_URL="${LLM_CRITIC_URL}/models"
if ! curl -fsS -m 5 "$LLM_CRITIC_HEALTH_URL" > /dev/null 2>&1; then
    echo "ERROR: LLM-critic vLLM-serve is not reachable at $LLM_CRITIC_HEALTH_URL" >&2
    echo "       Start the critic with scripts/gpt_oss_safeguard_vllm_serve.sh" >&2
    echo "       or export LLM_CRITIC_URL to point at the host where it runs." >&2
    exit 1
fi
echo "[health] LLM-critic vLLM-serve OK at $LLM_CRITIC_HEALTH_URL"

mkdir -p ${REPO_ROOT}/data_cache
mkdir -p ${LOG_DIR:-./logs}

cd ${REPO_ROOT}

# Defaults to the first four GPUs without overriding a value already set --
# see the note in scripts/gpt_oss_safeguard_vllm_serve.sh. The victim must
# see the same GPUs as the critic for the two to share memory.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" python3 -u evaluate.py \
        --exp_id $EXP_ID_NUM \
        --eval_model_code $EVAL_MODEL_CODE \
        --eval_dataset $EVAL_DATASET \
        --split $SPLIT \
        --model_name $MODEL_NAME \
        --top_k $TOP_K \
        --gpu_id $GPU_ID \
        --attack_method $ATTACK_METHOD \
        --adv_per_query $ADV_PER_QUERY \
        --score_function $SCORE_FUNCTION \
        --repeat_times $REPEAT_TIMES \
        --M $M \
        --seed $SEED \
        --log_name $LOG_NAME \
        --defend_method $DEFEND_METHOD \
        --removal_method $REMOVAL_METHOD \
        --lm_deploy_tp $LM_DEPLOY_TP \
        --lm_deploy_session_len $LM_DEPLOY_SESSION_LEN \
        --custom_attack_path $CUSTOM_ATTACK_PATH \
        --test_no_attack $TEST_NO_ATTACK \
        --attack_sample_num $ATTACK_SAMPLE_NUM \
        --query_omitted $QUERY_OMITTED \
        --llm_judge $LLM_JUDGE \
        --llm_judge_model $LLM_JUDGE_MODEL \
        --llm_critic_model "$LLM_CRITIC_MODEL" \
        --target_key_path $TARGET_QUERIES \
        2>&1 | tee ${LOG_DIR:-./logs}/$EXP_ID_NUM.txt
