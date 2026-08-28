#!/bin/bash

# Run it directly:
#   bash scripts/camodocs_token_manipulation.sh
#
# Or submit it to a scheduler. No #SBATCH directives are baked in: they
# cannot expand shell variables, so any partition/GPU names written here
# would hard-code one particular cluster. Pass your site's flags instead:
#   sbatch --partition=<your-partition> --gres=gpu:1 --time=24:00:00 \
#          -J camodocs_stage3 -o logs/out.%j scripts/camodocs_token_manipulation.sh
#
# Requires 1 GPU.

set -euo pipefail



: "${REPO_ROOT:?REPO_ROOT must be set to this repository (absolute path)}"
: "${DATA_ROOT:?DATA_ROOT must be set (writable; will hold Stage-3 output)}"

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-~/.cache/huggingface/datasets}"
export HF_TOKEN="${HF_TOKEN:-}"  # set via env if needed (gated model downloads)
export HF_HOME="${HF_HOME:-~/.cache/huggingface}"



EXP_ID=${EXP_ID:-camodocs-}   # output subdirectory prefix under $RESULT_PATH


# Dataset selection. All three datasets are supported by the bundled data
# (data_examples/, results/beir_results/, target_queries_fixed/).
EVAL_DATASET=${EVAL_DATASET:-hotpotqa}
case "$EVAL_DATASET" in
    hotpotqa|nq) SPLIT=test ;;
    msmarco)     SPLIT=train ;;
    *) echo "ERROR: unsupported EVAL_DATASET=$EVAL_DATASET"; exit 1 ;;
esac

CUSTOM_ATTACK_PATH=${REPO_ROOT}/data_examples/${EVAL_DATASET}_baseline_adv_answers.json
RESULT_PATH=${DATA_ROOT}/stage3_output/$EXP_ID/
# MS-MARCO targets are train-split queries, so it needs the train retrieval;
# the generic <dataset>-contriever.json does not contain them.
if [ "$EVAL_DATASET" = "msmarco" ]; then
    BENIGN_SCORE_PATH=${BENIGN_SCORE_PATH:-${REPO_ROOT}/results/beir_results/msmarco-contriever-train.json}
else
    BENIGN_SCORE_PATH=${BENIGN_SCORE_PATH:-${REPO_ROOT}/results/beir_results/${EVAL_DATASET}-contriever.json}
fi



SYNTH_DOCS_PATH=${REPO_ROOT}/data_examples/${EVAL_DATASET}_synth_benign.json

mkdir -p ${RESULT_PATH}
mkdir -p ${REPO_ROOT}/data_cache
mkdir -p ${LOG_DIR:-./logs}



# Pre-flight checks
if [ ! -f "$CUSTOM_ATTACK_PATH" ]; then
    echo "ERROR: gen_adv output not found at $CUSTOM_ATTACK_PATH"
    exit 1
fi


NUM_CAND=1000
NUM_ITER=30
ADV_PER_QUERY=5
DEBUG_MODE=False
DATA_NUM=${DATA_NUM:-2000}

# Optional data-parallel sharding: run N copies with MANUAL_RANK=0..N-1, then
# recombine with merge_shards.py. Defaults (-1) mean "no sharding".
MANUAL_WORLD_SIZE=${MANUAL_WORLD_SIZE:--1}
MANUAL_RANK=${MANUAL_RANK:--1}

# Coherence filter: GPT-2 PPL ranks the NUM_CAND HotFlip candidates and
# keeps the NUM_COHERENCE_CAND most fluent ones before the final
# candidate-by-candidate loss evaluation. Reduces high-PPL gibberish
# tokens that perplexity-based defenses would catch.
COHERENCE_FILTER=True
NUM_COHERENCE_CAND=100



if [ "$MANUAL_WORLD_SIZE" -eq -1 ]; then
    FILE_NAME=$EXP_ID-$EVAL_DATASET-synth_ance_mix_result.json
else
    FILE_NAME=$EXP_ID-$EVAL_DATASET-synth_ance_mix_result-$MANUAL_WORLD_SIZE-$MANUAL_RANK.json
fi

BREAK_NUM=2



# Do not silently overwrite a previous run's output.
if [ -f "${RESULT_PATH}/${FILE_NAME}" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "ERROR: ${RESULT_PATH}/${FILE_NAME} already exists. Set FORCE=1 to overwrite."
    exit 1
fi

# Pre-flight: every target query must appear in the retrieval file, otherwise
# mix_and_create_adv_result.py silently "[skip]"s it and produces nothing.
python3 - "$BENIGN_SCORE_PATH" "${REPO_ROOT}/target_queries_fixed/${EVAL_DATASET}_target_queries_fixed.json" <<'PYCHK'
import json,sys
bs=json.load(open(sys.argv[1])); tq=json.load(open(sys.argv[2]))
tq=list(tq) if isinstance(tq,list) else list(tq.keys())
miss=[q for q in tq if q not in bs]
if miss:
    print(f"ERROR: retrieval file covers only {len(tq)-len(miss)}/{len(tq)} target queries.")
    print(f"       {sys.argv[1]}")
       
    print("       Stage 3 would skip the uncovered queries. Set BENIGN_SCORE_PATH to the")
    print("       retrieval file matching this dataset/split.")
    sys.exit(1)
print(f"[pre-flight] retrieval covers all {len(tq)} target queries")
PYCHK

cd ${REPO_ROOT}/

python3 -u mix_and_create_adv_result.py \
            --synth_docs_path $SYNTH_DOCS_PATH \
            --custom_attack_path $CUSTOM_ATTACK_PATH \
            --result_path $RESULT_PATH \
            --file_name $FILE_NAME \
            --eval_dataset $EVAL_DATASET \
            --split $SPLIT \
            --num_cand $NUM_CAND \
            --num_iter $NUM_ITER \
            --adv_per_query $ADV_PER_QUERY \
            --debug_mode $DEBUG_MODE \
            --break_num $BREAK_NUM \
            --benign_score_path $BENIGN_SCORE_PATH \
            --coherence_filter $COHERENCE_FILTER \
            --num_coherence_cand $NUM_COHERENCE_CAND \
            --manual_world_size $MANUAL_WORLD_SIZE \
            --manual_rank $MANUAL_RANK \
            --data_num $DATA_NUM 2>&1 | tee ${LOG_DIR:-./logs}/$EXP_ID-create-adv-result-$MANUAL_RANK.txt
