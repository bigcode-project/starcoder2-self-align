# Reproduce the experiments

> [!IMPORTANT]
> **General requirements**
>
> Before you start, make sure you cloned the respository. Make sure you installed the required packages with `pip install -r requirements.txt`. Different package versions may impact the reproducibility of the results.
>
> We obtained the results with the subsequent hardware and environment:
>
> - One NVIDIA A100 80G GPU
> - Python 3.10.0
>
> In case you face issues, we provide the raw outputs we generated in the [evalplus_results](evalplus_results) directory.

## Reproduce HumanEval(+) and MBPP(+)

We pack multiple problems into one batch to speed up the inference. A different batch size may lead to slightly worse/better results due to the floating point round off resulted from the underlying [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) optimization.

Make sure you set `CUDA_VISIBLE_DEVICES` to the GPU you want to use and `cd`ed to the root directory of the repo. We assume you use device 0 in the following commands.

### HumanEval(+)

```bash
MODEL_KEY=bigcode/starcoder2-15b-instruct-v0.1
MODEL=bigcode/starcoder2-15b-instruct-v0.1
DATASET=humaneval
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python -m evaluation.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 16 \
  --n_samples_per_problem 1 \
  --n_batches 1

python -m evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
# humaneval (base tests)
# pass@1: 0.726
# humaneval+ (base + extra tests)
# pass@1: 0.634
```

### MBPP(+)

```bash
MODEL_KEY=bigcode/starcoder2-15b-instruct-v0.1
MODEL=bigcode/starcoder2-15b-instruct-v0.1
DATASET=mbpp
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl

CUDA_VISIBLE_DEVICES=0 python -m evaluation.text2code \
  --model_key $MODEL_KEY \
  --model_name_or_path $MODEL \
  --save_path $SAVE_PATH \
  --dataset $DATASET \
  --temperature 0.0 \
  --top_p 1.0 \
  --max_new_tokens 512 \
  --n_problems_per_batch 16 \
  --n_samples_per_problem 1 \
  --n_batches 1

python -m evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
# mbpp (base tests)
# pass@1: 0.642
# mbpp+ (base + extra tests)
# pass@1: 0.526
```