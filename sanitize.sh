#!/bin/bash
set -e

SOURCE=$1
TARGET=$2

echo "Sanitizing.."
python -m star_align.sanitize_data \
    --data_files $SOURCE \
    --output_file $TARGET \
    --parse_raw_response True \
    --exact_match_dedup True \
    --passing_only True \
    --include_left_failed False

if [[ -n $DECONTAMINATION ]]; then
    echo "Decontaminating.. (saving to decontamination-output)"
    python -m star_align.decontamination.find_substrings \
        --dataset_name "json" \
        --output_file $TARGET \
        --output_dir decontamination-output \
        --columns instruction response \
        --data_files $TARGET
fi

echo "Minihash dedup.."
python -m star_align.minhash_dedup \
    --data_files $TARGET \
    --column instruction \
    --output $TARGET

python -m star_align.minhash_dedup \
    --data_files $TARGET \
    --column response \
    --output $TARGET

python -m star_align.minhash_dedup \
    --data_files $TARGET \
    --column code_representation \
    --ignore_empty True \
    --output $TARGET
