import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict, cast
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from star_align.llm_wrapper import GenerationConfig, get_model_context
from star_align.prompt_template import SC2_INSTRUCT_PROMPT as PROMPT_TEMPLATE
from star_align.utils import chunked

class Text2CodeProblem(TypedDict):
    id: str
    instruction: str
    response_prefix: str


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())

def map_mbpp_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    start_index = prompt.index('"""')
    end_index = prompt.rindex('"""')
    prompt = prompt[start_index + 3 : end_index]
    assert_index = prompt.index("assert")
    instruction = prompt[:assert_index].strip()
    if not instruction.endswith("."):
        instruction += "."
    assertion = prompt[assert_index:].strip()
    instruction = f"""{instruction}

Your code should pass the following assertion:
```python
{assertion}
```"""
    prefix = "" if PROMPT_TEMPLATE.endswith("\n") else "\n"
    response_prefix = f"""{prefix}```python"""
    return Text2CodeProblem(
        id=str(id), instruction=instruction, response_prefix=response_prefix
    )


def map_humaneval_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    prompt_header = "Write a Python function to solve the given task:"
    instruction = f"""{prompt_header}
```python
{prompt}
```"""
    prefix = "" if PROMPT_TEMPLATE.endswith("\n") else "\n"
    prefix_template = "```python\n{prompt}"
    response_prefix = prefix + (
        prefix_template.replace("{prompt}", prompt)
        if "{prompt}" in prefix_template
        else prefix_template
    )
    return Text2CodeProblem(
        id=id, instruction=instruction, response_prefix=response_prefix
    )


@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal["humaneval", "mbpp"]
    save_path: str

    n_batches: int
    n_problems_per_batch: int
    n_samples_per_problem: int

    model_name_or_path: str | None = None


def main():
    parser = HfArgumentParser((Args, GenerationConfig))
    args, generation_config = cast(
        tuple[Args, GenerationConfig],
        parser.parse_args_into_dataclasses(),
    )
    raw_problem_fn, map_problem_fn = (
        (get_humaneval_raw_problems, map_humaneval_problem)
        if args.dataset == "humaneval"
        else (get_mbpp_raw_problems, map_mbpp_problem)
    )
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    state = get_model_context(args.model_key, args.model_name_or_path)

    problems_chunked = list(chunked(list(problems), args.n_problems_per_batch))
    iter = itertools.product(problems_chunked, range(args.n_batches))
    n_total = len(problems_chunked) * args.n_batches

    Path(args.save_path).write_text("")
    for problems, batch_idx in tqdm(iter, total=n_total):
        task_ids = [problem["id"] for problem in problems]
        prompts = [
            # TODO: make it generic for all models
            PROMPT_TEMPLATE.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
            for problem in problems
        ]
        print("PROMPT")
        print(prompts[-1])
        all_prompts = prompts * args.n_samples_per_problem
        all_task_ids = task_ids * args.n_samples_per_problem
        response = state.complete(generation_config, all_prompts, stop_tokens=["\n```"])
        completions = response.decoded_outputs
        assert len(problems) <= args.n_problems_per_batch
        assert len(completions) == len(problems) * args.n_samples_per_problem
        print("COMPLETION")
        print(completions[-1])
        samples = [
            dict(
                task_id=task_id,
                completion=completion[
                    : (
                        index
                        if (index := completion.find("```")) != -1
                        else len(completion)
                    )
                ],
            )
            for task_id, completion in zip(all_task_ids, completions)
        ]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
