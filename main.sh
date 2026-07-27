#!/bin/bash

############################
# Configuration
############################

cd OpenIKLR/

DATASET="FOLIO"

MODEL=""
API_URL=""
API_KEY=""

K_MAX=0

############################
# Helper functions
############################

run_llm () {
    python call_llm.py \
        --input_file "$1" \
        --output_file "$2" \
        --api_key "$API_KEY" \
        --api_base_url "$API_URL" \
        --model "$MODEL" \
        --task_type "$3"
}


run_solver () {
    python call_solver.py \
        --input_file "$1" \
        --output_file "$2"
}


############################
# Translation
############################

run_llm \
    "data/${DATASET}.json" \
    "${MODEL}/translation/${DATASET}.json" \
    "translation"


run_solver \
    "${MODEL}/translation/${DATASET}.json" \
    "${MODEL}/solver/${DATASET}.json"


############################
# Iterative Logic Completion
############################

CURRENT_INPUT="${MODEL}/solver/${DATASET}_invalid.json"


for ((k=1; k<=K_MAX; k++))
do

    if [ "$k" -eq 1 ]; then
        TASK="logic_completion_init"
    else
        TASK="logic_completion"
    fi


    COMPLETION_FILE="${MODEL}/completion/${DATASET}_${k}.json"

    SOLVER_FILE="${MODEL}/solver/${DATASET}_${k}.json"


    echo "Running logic completion step ${k}/${K_MAX}"


    run_llm \
        "$CURRENT_INPUT" \
        "$COMPLETION_FILE" \
        "$TASK"


    run_solver \
        "$COMPLETION_FILE" \
        "$SOLVER_FILE"

    CURRENT_INPUT="${MODEL}/solver/${DATASET}_${k}_invalid.json"

done



############################
# Merge valid data
############################

python tools.py \
    --input_file "${MODEL}/solver" \
    --output_file "${MODEL}/solver/${DATASET}_merge_valid.json" \
    --dataset "${DATASET}" \
    --task_type "merge_valid_data"
    


############################
# Fact Verification
############################

run_llm \
    "${MODEL}/solver/${DATASET}_merge_valid.json" \
    "${MODEL}/sl2nl/${DATASET}_merge_valid.json" \
    "fact_verification"

CURRENT_INPUT="${MODEL}/sl2nl/${DATASET}_merge_valid_notfact.json"


############################
# Iterative Fact Completion
############################

for ((k=1; k<=K_MAX; k++))
do

    if [ "$k" -eq 1 ]; then
        TASK="fact_completion_init"
    else
        TASK="fact_completion"
    fi


    COMPLETION_FILE="${MODEL}/completion/${DATASET}_merge_valid_notfact_${k}.json"

    SOLVER_FILE="${MODEL}/solver/${DATASET}_merge_valid_notfact_${k}.json"

    VERIFY_FILE="${MODEL}/sl2nl/${DATASET}_merge_valid_notfact_${k}_valid.json"

    echo "Running fact completion step ${k}/${K_MAX}"


    run_llm \
        "$CURRENT_INPUT" \
        "$COMPLETION_FILE" \
        "$TASK"


    run_solver \
        "$COMPLETION_FILE" \
        "$SOLVER_FILE"


    run_llm \
        "${SOLVER_FILE%.*}_valid.json" \
        "$VERIFY_FILE" \
        "fact_verification"


    CURRENT_INPUT="${VERIFY_FILE%.*}_notfact.json"

done


############################
# Final processing
############################

python tools.py \
    --input_file "${MODEL}" \
    --output_file "${MODEL}/final_results/${DATASET}_final.json" \
    --dataset "${DATASET}" \
    --api_key "$API_KEY" \
    --api_base_url "$API_URL" \
    --model "$MODEL" \
    --task_type "get_final_data"


python tools.py \
    --input_file "${MODEL}/final_results/${DATASET}_final.json" \
    --task_type "evaluation"


echo "Finished."