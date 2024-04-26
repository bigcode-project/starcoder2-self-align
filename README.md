# StarCoder2-Instruct: Self-Aligned, Transparent, and Fully Permissive

> [!WARNING]
> This documentation is still WIP.

## Data generation pipeline

We used VLLM's [OpenAI compatible server](https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html) for data generation. So, before running the following commands, make sure the VLLM server is running, and the associated `openai` environment variables are set.

For example, you can start an VLLM server with `docker`:

```shell
docker run --gpus '"device=0"' \
    -v $HF_HOME:/root/.cache/huggingface \                            
    -p 10000:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.3.3 \
    --model bigcode/starcoder2-15b \
    --tensor-parallel-size 1 --dtype bfloat16
```

And then set the environment variables as follows:

```shell
export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://localhost:10000/v1/"
```

### Snippet to concept

```shell
python src/star_align/self_ossinstruct.py \
    --instruct_mode "S->C" \
    --seed_data_files /path/to/seeds.jsonl \
    --max_new_data 50000 \
    --tag concept_gen \
    --temperature 0.7 \
    --seed_code_start_index 0 \
    --model bigcode/starcoder2-15b \
    --num_fewshots 8 \
    --num_batched_requests 32 \
    --num_sample_per_request 1
```

### Concept to instruction

```shell
python src/star_align/self_ossinstruct.py \
    --instruct_mode "C->I" \
    --seed_data_files /path/to/concepts.jsonl \
    --max_new_data 50000 \
    --tag instruction_gen \
    --temperature 0.7 \
    --seed_code_start_index 0 \
    --model bigcode/starcoder2-15b \
    --num_fewshots 8 \
    --num_sample_per_request 1 \
    --num_batched_request 32
```

### Instruction to response w/ self-validation code

```shell
python src/star_align/self_ossinstruct.py \
    --instruct_mode "I->R" \
    --seed_data_files path/to/instructions.jsonl  \
    --max_new_data 50000 \
    --tag response_gen \
    --seed_code_start_index 0 \
    --model bigcode/starcoder2-15b \
    --num_fewshots 1 \
    --num_batched_request 8 \
    --num_sample_per_request 10 \
    --temperature 0.7
```

### Execution filter

> [!WARNING]
> Though we implemented reliability guards, it is highly recommended to run execution in a sandbox environment. The command below doesn't provide sandboxing by default.

```shell
python src/star_align/execution_filter.py --response_path /path/to/response.jsonl --result_path /path/to/filtered.jsonl
# The current implementation may cause deadlock.
# If you encounter deadlock, manually do `ps -ef | grep execution_filter` and kill the stuck process.
# Note that filtered.jsonl may contain multiple passing samples for the same instruction which needs further selection.
```

### Data sanitization and selection

```shell
RAW=1 python src/star_align/sanitize_data.py /path/to/filtered.jsonl /path/to/sanitized.jsonl
python src/star_align/clean_data.py --data_files /path/to/sanitized.jsonl --output_file /path/to/sanitized.jsonl --diversify_func_names
SMART=1 python src/star_align/sanitize_data.py /path/to/sanitized.jsonl /path/to/sanitized.jsonl
```
