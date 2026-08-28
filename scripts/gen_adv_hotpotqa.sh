#!/bin/bash
# Generate adversarial drafts for HotpotQA via gen_adv.py + gpt-4o-mini.
# Stage 2 of the CamoDocs pipeline (output feeds mix_and_create_adv_result.py).
#
# BEIR HotpotQA downloads automatically on first run, into
# ${REPO_ROOT}/datasets/hotpotqa/. gen_adv.py loads the BEIR corpus before it
# reads queries.jsonl, so the file is in place by the time it is needed.

set -euo pipefail



: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"
: "${REPO_ROOT:?REPO_ROOT must be set to this repository (absolute path)}"
: "${DATA_ROOT:?DATA_ROOT must be set (writable; will hold Stage-2 output)}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export HF_TOKEN="${HF_TOKEN:-}"
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"

cd ${REPO_ROOT}/

EVAL_DATASET=hotpotqa
SPLIT=test
MODEL_NAME=gpt4o_mini
ADV_PER_QUERY=5             # n_draft = 5 adversarial drafts per target query
TARGET_QUERIES=${REPO_ROOT}/target_queries_fixed/${EVAL_DATASET}_target_queries_fixed.json

# BEIR HotpotQA inputs
QUERIES_JSONL_PATH=${REPO_ROOT}/datasets/hotpotqa/queries.jsonl

# Stage 2 output
SAVE_PATH=${DATA_ROOT}/stage2_output
FILE_NAME=hotpotqa_with_queries_answer

mkdir -p "$SAVE_PATH"
mkdir -p "${LOG_DIR:-./logs}"

python3 -u gen_adv.py \
        --eval_dataset       $EVAL_DATASET \
        --split              $SPLIT \
        --model_name         $MODEL_NAME \
        --adv_per_query      $ADV_PER_QUERY \
        --target_queries_path  $TARGET_QUERIES \
        --save_path          $SAVE_PATH \
        --queries_jsonl_path $QUERIES_JSONL_PATH \
        --file_name          $FILE_NAME \
        2>&1 | tee ${LOG_DIR:-./logs}/gen_adv_hotpotqa.txt
