#!/bin/bash

# Run it directly:
#   bash scripts/eval_trustrag.sh
#
# Or submit it to a scheduler. No #SBATCH directives are baked in: they
# cannot expand shell variables, so any partition/GPU names written here
# would hard-code one particular cluster. Pass your site's flags instead:
#   sbatch --partition=<your-partition> --gres=gpu:1 --time=24:00:00 \
#          -J trustrag -o logs/out.%j scripts/eval_trustrag.sh
#
# Requires 1 GPU.

# Main-table defense — TrustRAG (kmeans_ngram removal + conflict-resolution
# prompting) vs. CamoDocs. HotpotQA, Llama-3.1-8B victim, contriever retriever.
# Fills the TrustRAG row of the main defense table.

set -euo pipefail





: "${REPO_ROOT:?REPO_ROOT must be set to this repository (absolute path)}"
: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set (LLM judge)}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export HF_TOKEN="${HF_TOKEN:-}"  # set via env if needed (gated model downloads)
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

EXP_ID_NUM=${EXP_ID_NUM:-eval_trustrag-}

EVAL_MODEL_CODE="contriever"
EVAL_DATASET="${EVAL_DATASET:-hotpotqa}"
SPLIT="${SPLIT:-test}"
TOP_K="5"
GPU_ID="0"
SCORE_FUNCTION="dot"

TEST_NO_ATTACK=False
ATTACK_METHOD="LM_targeted"
DEFEND_METHOD="conflict"
REMOVAL_METHOD="kmeans_ngram"

M="100"
REPEAT_TIMES="10"
SEED="12"
ADV_PER_QUERY=10
NOTE="None"

MODEL_NAME="llama"
MODEL_NAME_FOR_LOG="${MODEL_NAME//\//-}"
LM_DEPLOY_TP=1
LM_DEPLOY_SESSION_LEN=131072

ATTACK_SAMPLE_NUM=1000

LLM_JUDGE=True
LLM_JUDGE_MODEL=gpt-4.1-mini-2025-04-14

LOG_NAME=$EXP_ID_NUM-$EVAL_DATASET-$EVAL_MODEL_CODE-$MODEL_NAME_FOR_LOG-camodocs-trustrag-M$M-rep$REPEAT_TIMES-$NOTE.log

# CamoDocs adversarial documents (merged Stage-3 output).
CUSTOM_ATTACK_PATH=${CUSTOM_ATTACK_PATH:-${REPO_ROOT}/data_examples/camodocs_${EVAL_DATASET}_adv_text_merged.json}
TARGET_QUERIES=${TARGET_QUERIES:-${REPO_ROOT}/target_queries_fixed/${EVAL_DATASET}_target_queries_fixed.json}

QUERY_OMITTED=${QUERY_OMITTED:-True}   # set False to reproduce the PoisonedRAG baseline

mkdir -p ${REPO_ROOT}/data_cache
mkdir -p ${LOG_DIR:-./logs}

cd ${REPO_ROOT}/

python3 -u evaluate.py \
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
        --target_key_path $TARGET_QUERIES \
        2>&1 | tee ${LOG_DIR:-./logs}/$EXP_ID_NUM.txt
