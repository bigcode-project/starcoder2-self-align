import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypedDict, cast
from evalplus.data import get_human_eval_plus, get_mbpp_plus, write_jsonl

from evoeval.data import get_evo_eval
from transformers import HfArgumentParser

from star_align.utils import infer_prompt_template, is_base_model

from vllm import LLM, SamplingParams


class Text2CodeProblem(TypedDict):
    id: str
    prompt: str
    instruction: str
    response_prefix: str


# MBPP_INSTRUCTION = """{nl_description} Your code should satisfy the following assertion:
# ```python
# {assertions}
# ```
# Enclose your solution in ```python and ```"""


def get_mbpp_raw_problems() -> list[dict]:
    problems = get_mbpp_plus()
    return list(problems.values())


def get_humaneval_raw_problems() -> list[dict]:
    problems = get_human_eval_plus()
    return list(problems.values())


def get_evoeval_raw_problems(dataset: str):
    def get_raw_problems() -> list[dict]:
        problems = get_evo_eval(dataset)
        return list(problems.values())

    return get_raw_problems


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

```python
{assertion}
```"""
    prefix = ""
    response_prefix = f"""{prefix}```python"""
    return Text2CodeProblem(
        id=str(id),
        prompt=prompt,
        instruction=instruction,
        response_prefix=response_prefix,
    )


def map_humaneval_problem(p: dict) -> Text2CodeProblem:
    id = p["task_id"]
    prompt = p["prompt"]
    prompt = prompt.strip()
    # try:
    #     docstring_index = prompt.index('"""')
    # except ValueError:
    #     docstring_index = prompt.index("'''")
    # signature = prompt[:docstring_index].strip()
    # Instruction
    # instruction = f"""Complete the implementation of the following function:
    prompt_header = os.getenv(
        "PROMPT_HEADER", "Write a Python function to solve the following task:"
    )
    instruction = f"""{prompt_header}
```python
{prompt}
```"""
    prefix = ""
    prefix_template = os.getenv("PREFIX_TEMPLATE", "```python")
    response_prefix = prefix + (
        prefix_template.replace("{prompt}", prompt)
        if "{prompt}" in prefix_template
        else prefix_template
    )
    # response_prefix = f"""{prefix}```python
    # {prompt}"""
    return Text2CodeProblem(
        id=id,
        prompt=prompt,
        instruction=instruction,
        response_prefix=response_prefix,
    )


@dataclass(frozen=True)
class Args:
    model_key: str
    dataset: Literal[
        "humaneval",
        "mbpp",
        "EvoEval_difficult",
        "EvoEval_creative",
        "EvoEval_subtle",
        "EvoEval_combine",
        "EvoEval_tool_use",
        "EvoEval_verbose",
        "EvoEval_concise",
    ]
    save_path: str
    n_samples_per_problem: int = field(default=1)
    max_new_tokens: int = field(default=1024)
    top_p: float = field(default=1.0)
    temperature: float = field(default=0.0)
    model_name_or_path: str | None = None


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    raw_problem_fn, map_problem_fn = (
        (get_evoeval_raw_problems(args.dataset), map_humaneval_problem)
        if args.dataset.startswith("EvoEval_")
        else (
            (get_humaneval_raw_problems, map_humaneval_problem)
            if args.dataset == "humaneval"
            else (get_mbpp_raw_problems, map_mbpp_problem)
        )
    )
    raw_problems = raw_problem_fn()
    problems = list(map(map_problem_fn, raw_problems))

    engine = LLM(
        tokenizer=args.model_key, model=args.model_name_or_path or args.model_key
    )

    base_model_prompt = is_base_model(args.model_key)

    stop: str | list[str] = (
        "\n```\n"
        if not base_model_prompt
        else ["\ndef ", "\nclass ", "\nimport ", "\nfrom ", "\nassert ", "\n# "]
    )
    sampling_params = SamplingParams(
        n=args.n_samples_per_problem,
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_k=-1,
        top_p=args.top_p,
        stop=stop,
    )

    if base_model_prompt:
        print("Base model")
    else:
        prompt_template = infer_prompt_template(
            os.getenv("TOKENIZER") or args.model_name_or_path or args.model_key
        )
        # prompt_template = PROMPT_TEMPLATE
        print("Using:", prompt_template)

    prompts: list[str] = []
    for problem in problems:
        if not base_model_prompt:
            prompt = prompt_template.format(
                instruction=problem["instruction"], response=problem["response_prefix"]
            )
        else:
            prompt = problem["prompt"]
        prompts.append(prompt)

    results = engine.generate(prompts, sampling_params)
    Path(args.save_path).write_text("")

    step = 20
    print_or_not = [idx == 0 or idx % step == 0 for idx in range(len(problems))]

    def sanitize(output: str) -> str:
        if not base_model_prompt:
            return output.split("```python")[-1].split("```")[0]
        for s in stop:
            output = output.rsplit(s, 1)[0]
        return output

    for problem, prompt, result, print_debug in zip(
        problems, prompts, results, print_or_not
    ):
        if print_debug:
            print("[Example Prompt]")
            print(prompt)
            print("[Example Completion]")
            print(result.outputs[0].text)
        samples = [
            dict(
                task_id=problem["id"],
                completion=sanitize(output.text),
            )
            for output in result.outputs
        ]
        write_jsonl(args.save_path, samples, append=True)


if __name__ == "__main__":
    main()
