#!/bin/bash

echo "MODE: $MODE"
echo "SEED_DATA_FILE: $SEED_DATA_FILE"
echo "INDEX: $INDEX"
echo "MAX_NEW_DATA: $MAX_NEW_DATA"
echo "DIR: $1"

# if mode is "I->R", num of samples is 10, otherwise 1
if [[ "$MODE" == "I->R" ]]; then
    N_SAMPLES=1
    NUM_FEWSHOTS=1
    NUM_BATCHED_REQUESTS=4096
    ASYNC_MICRO_BATCH_SIZE=16
else
    N_SAMPLES=1
    NUM_FEWSHOTS=8
    NUM_BATCHED_REQUESTS=4096
    ASYNC_MICRO_BATCH_SIZE=8
fi

echo "N_SAMPLES: $N_SAMPLES"
echo "NUM_FEWSHOTS: $NUM_FEWSHOTS"
echo "NUM_BATCHED_REQUESTS: $NUM_BATCHED_REQUESTS"
echo "ASYNC_MICRO_BATCH_SIZE: $ASYNC_MICRO_BATCH_SIZE"

COMMAND="python -m star_align.self_ossinstruct \
    --async_micro_batch_size $ASYNC_MICRO_BATCH_SIZE \
    --use_vllm_server True \
    --instruct_mode '$MODE' \
    --seed_data_files $SEED_DATA_FILE \
    --max_new_data $MAX_NEW_DATA \
    --tag sc2-${NUM_FEWSHOTS}shot \
    --temperature 0.7 \
    --seed_code_start_index $INDEX \
    --model bigcode/starcoder2-15b \
    --num_fewshots $NUM_FEWSHOTS \
    --num_batched_requests $NUM_BATCHED_REQUESTS \
    --num_sample_per_request $N_SAMPLES \
    --save_dir $1"

if [[ -n "$2" ]]; then
    COMMAND="$COMMAND --continue_from $2"
fi

echo "Running command: $COMMAND"
eval $COMMAND
