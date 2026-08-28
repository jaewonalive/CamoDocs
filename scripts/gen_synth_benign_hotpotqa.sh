#!/bin/bash
# Stage 1: Generate synthesized benign-looking drafts for HotpotQA using
# gpt-4o-mini via the OpenAI API. The benign drafts are the "carrier"
# side of CamoDocs's sub-document merging (later optimized via Stage 3
# token replacement and concatenated with adversarial sub-documents).
#
# Runs concurrently (async, --concurrency in-flight) — 1000 queries
# typically finishes in ~3-5 minutes wall-clock.
#
# Self-contained mode: --target_queries_path carries 'correct answer'
# per qid, so no --queries_jsonl_path is needed; the loader detects the
# dict shape and uses 'correct answer' directly.

set -euo pipefail



: "${OPENAI_API_KEY:?OPENAI_API_KEY must be set}"   # read from the environment by gen_synth_benign.py;
                                                    # NOT passed as --api_key, which would expose it in `ps`
: "${REPO_ROOT:?REPO_ROOT must be set to this repository (absolute path)}"
: "${DATA_ROOT:?DATA_ROOT must be set (writable; will hold Stage-1 output)}"

cd ${REPO_ROOT}/

MODEL=gpt-4o-mini-2024-07-18
CONCURRENCY=5
ADV_PER_QUERY=5            # n_draft = 5 benign drafts per target query

TARGET_QUERIES_PATH=${REPO_ROOT}/data_examples/hotpotqa_baseline_adv_answers.json
SAVE_PATH=${DATA_ROOT}/stage1_output
FILE_NAME=hotpotqa_synth_benign

mkdir -p "$SAVE_PATH"
mkdir -p "${LOG_DIR:-./logs}"

python3 -u gen_synth_benign.py \
        --eval_dataset        hotpotqa \
        --target_queries_path $TARGET_QUERIES_PATH \
        --save_path           $SAVE_PATH \
        --file_name           $FILE_NAME \
        --model_name          $MODEL \
        --adv_per_query       $ADV_PER_QUERY \
        --concurrency         $CONCURRENCY \
        2>&1 | tee ${LOG_DIR:-./logs}/gen_synth_benign_hotpotqa.txt
