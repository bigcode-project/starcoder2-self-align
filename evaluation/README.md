# Evaluation

> [!IMPORTANT]
> **General requirements**
>
> Before you start, make sure you have cloned the repository and you are in the **root directory of the project**. Make sure you installed the required packages with `pip install -e .`. Different package versions may impact the reproducibility of the results.

## Running EvalPlus with vLLM

We implemented batched inference in [evaluation/text2code_vllm.py] using [vLLM](https://docs.vllm.ai/en/latest/). This speed up the evaluation significantly: **a greedy decoding run can be finished within 20 seconds**. Here is the command:

```bash
MODEL=/path/to/your/model
DATASET=humaneval # or mbpp
SAVE_PATH=evalplus-$(basename $MODEL)-$DATASET.jsonl
CUDA_VISIBLE_DEVICES=0 python -m evaluation.text2code_vllm \
    --model_key $MODEL \
    --dataset $DATASET \
    --save_path $SAVE_PATH

python -m evalplus.evaluate --dataset $DATASET --samples $SAVE_PATH
```

## Reproduce StarCoder2-Instruct

> [!NOTE]
>
> We obtained the results with the subsequent hardware and environment:
>
> - One NVIDIA A100 80G GPU
> - Python 3.10.0
>
> In case you face issues, we provide the raw outputs we generated in the [evalplus_results](evalplus_results) directory.

### Reproduce HumanEval(+) and MBPP(+)

We pack multiple problems into one batch to speed up the inference. A different batch size may lead to slightly worse/better results due to the floating point round off resulted from the underlying [cuBLAS](https://docs.nvidia.com/cuda/cublas/index.html) optimization.

Make sure you set `CUDA_VISIBLE_DEVICES` to the GPU you want to use and `cd`ed to the root directory of the repo. We assume you use device 0 in the following commands.

#### HumanEval(+)

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

#### MBPP(+)

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
