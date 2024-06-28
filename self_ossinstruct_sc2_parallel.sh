#!/bin/bash

echo "MODE: $MODE"
echo "SEED_DATA_FILE: $SEED_DATA_FILE"
echo "INDEX: $INDEX"
echo "MAX_NEW_DATA: $MAX_NEW_DATA"
echo "DIR: $1"

NUM_GPUS=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits | head -n 1)

DATA_CHUNK_SIZE=$(($MAX_NEW_DATA / $NUM_GPUS))
REMAINDER=$(($MAX_NEW_DATA % $NUM_GPUS))

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

PIDS=()
function killall_pids {
    for pid in ${PIDS[@]}; do
        kill $pid
    done
}
trap killall_pids SIGINT SIGTERM

for (( GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++ ))
do
    START_INDEX=$(($INDEX + $GPU_ID * $DATA_CHUNK_SIZE))
    if [[ $GPU_ID -lt $REMAINDER ]]; then
        CHUNK_SIZE=$(($DATA_CHUNK_SIZE + 1))
    else
        CHUNK_SIZE=$DATA_CHUNK_SIZE
    fi
    END_INDEX=$(($START_INDEX + $CHUNK_SIZE - 1))

    echo "Starting process for GPU $GPU_ID with data from $START_INDEX to $END_INDEX..."

    OUTDIR="$1/$GPU_ID"
    mkdir -p $OUTDIR

    CUDA_VISIBLE_DEVICES=$GPU_ID python -m star_align.self_ossinstruct \
        --async_micro_batch_size $ASYNC_MICRO_BATCH_SIZE \
        --use_vllm_server False \
        --instruct_mode "$MODE" \
        --seed_data_files $SEED_DATA_FILE \
        --max_new_data $CHUNK_SIZE \
        --tag sc2-${NUM_FEWSHOTS}shot \
        --temperature 0.7 \
        --seed_code_start_index $START_INDEX \
        --model bigcode/starcoder2-15b \
        --num_fewshots $NUM_FEWSHOTS \
        --num_batched_requests $NUM_BATCHED_REQUESTS \
        --num_sample_per_request $N_SAMPLES \
        --save_dir $OUTDIR &
    PIDS+=($!)
done

wait

if [[ $? -ne 0 ]]; then
    echo "Error in one of the processes. Exiting... Check logs for more details."
    exit 1
fi

# dir for final res
FINAL="$1/final"
FINAL_FILE="$FINAL/aggregated-${MODE}.jsonl"

echo "All processes finished. Aggregating results... to $FINAL_FILE"

# aggregate!
mkdir -p $FINAL
touch $FINAL_FILE

for (( GPU_ID=0; GPU_ID<$NUM_GPUS; GPU_ID++ ))
do
  # get first file for dir
  FILE=$(ls $1/$GPU_ID | head -n 1)
  cat $1/$GPU_ID/$FILE >> $FINAL_FILE
done

echo "Done!"
