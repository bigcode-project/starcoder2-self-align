"""Deduplication, filtering, and selection"""

import random
import os
import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from star_align.utils import find_code_blocks, write_jsonl


@dataclass(frozen=True)
class Args:
    data_files: list[str]
    output_file: str
    shuffle: bool = field(default=True)
    parse_raw_response: bool = field(default=True)
    passing_only: bool = field(default=True)
    data_augmentation: bool = field(default=False)
    exact_match_dedup: bool = field(default=True)
    n_cores: int = field(default=os.cpu_count() or 1)
    diversify_func_names: bool = field(default=True)
    minihash_dedup: bool = field(default=False)
    seed: int = field(default=6666)


def extract_and_concat_function_names(python_content):
    """
    Extracts all function names from a given Python content string and concatenates them into a single string.

    Parameters:
    - python_content: A string containing the Python code to analyze.

    Returns:
    - A string containing all function names defined in the content, concatenated.
    """
    tree = ast.parse(python_content)
    function_names: list[str] = []

    # Define a node visitor that adds the name of each function definition it visits
    class FunctionDefVisitor(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            function_names.append(node.name)
            # Process the subtree for this node
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            function_names.append(node.name)
            self.generic_visit(node)

    # Create a node visitor and walk through the AST
    visitor = FunctionDefVisitor()
    visitor.visit(tree)

    # Concatenate all function names into a single string
    return " ".join(function_names)


INCOMPLETE_SUBSTRINGS = [
    "todo",
    "fixme",
    "write your code here",
    "your code here",
    "your code goes here",
    "notimplemented",
]

RESPONSE_TEST_SPLIT = "</response>\n\n<tests>"


def preprocess_and_filter(x: dict) -> dict:
    """Filter out responses with wrong format"""

    def wrong_format(x: dict) -> dict:
        return {k: v for k, v in x.items()} | dict(wrong_format=True)

    response: str = x["response"]
    if RESPONSE_TEST_SPLIT not in response:
        return wrong_format(x)
    if any(substring in response.lower() for substring in INCOMPLETE_SUBSTRINGS):
        return wrong_format(x)
    splits = response.split(RESPONSE_TEST_SPLIT)
    if len(splits) != 2:
        return wrong_format(x)
    response, tests = cast(tuple[str, str], tuple(map(str.strip, splits)))
    response_codeblocks = find_code_blocks(response, "python")
    tests_codeblocks = find_code_blocks(tests, "python")
    if len(response_codeblocks) == 0 or len(tests_codeblocks) == 0:
        return wrong_format(x)

    tests_content = "\n".join(tests_codeblocks)
    if "assert" not in tests or all(
        l.startswith("def")
        or l.startswith("class")
        or l.startswith("import")
        or l.startswith("from")
        for l in tests_content.splitlines()
        if len(l) > 0 and l[0].isalpha()
    ):
        return wrong_format(x)

    newx = {k: v for k, v in x.items() if k != "response"} | dict(
        response=response, tests=tests
    )
    return newx


def augment_data(x: dict, index: int) -> dict:
    random.seed(index)
    tests_content = "\n".join(find_code_blocks(x["tests"]))
    lines = tests_content.splitlines()
    if all(l.startswith("assert") for l in lines):
        ks = [1, 2, 3, 4, 5]
        assertions = random.sample(lines, k=min(random.choice(ks), len(lines)))
        assertion = "\n".join(assertions)
        assertion_term = "assertion" + ("s" if len(assertions) > 1 else "")
    else:
        assertion = tests_content
        assertion_term = "test case"
    if (
        "assert" in assertion
        # 5 lines augmented block max
        and len(assertion.splitlines()) <= 5
        and random.random() < 0.5
        and "assert" not in x["instruction"]
        and "for example" not in x["instruction"].lower()
        and "test" not in x["instruction"].lower()
    ):
        assert "assert" in assertion
        assertion_str = (
            f"Your code should pass the following {assertion_term}:\n```python\n"
            + assertion.strip()
            + "\n```"
        )
        new_instruction = f"{x['instruction']}\n\n{assertion_str}"
        newx = {k: v for k, v in x.items()} | dict(instruction=new_instruction)
        return newx
    return x


# raw response -> response + test
# response/test -> passing (opt: passing)
# (not)passing -> unique
# unique -> aug / minihash / selection / educational -> final


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])

    raw_data = load_dataset("json", data_files=args.data_files, split="train")
    print("Raw samples:", len(raw_data))

    if args.parse_raw_response:
        raw_data = raw_data.map(preprocess_and_filter, num_proc=args.n_cores)
        raw_data = raw_data.filter(
            lambda x: not x["wrong_format"], num_proc=args.n_cores
        )
        print("Correct format:", len(raw_data))

    if args.passing_only:
        raw_data = raw_data.filter(lambda x: x["pass"], num_proc=args.n_cores)
        print("Passing only:", len(raw_data))

    if args.shuffle:
        raw_data = raw_data.shuffle(seed=args.seed)

    def mk_key(instruction: str) -> str:
        return "".join(instruction.split())

    seen_keys = set[str]()
    if args.exact_match_dedup:
        new_data = list[dict]()
        for d in tqdm(raw_data):
            key_i, key_r = mk_key(d["instruction"]), mk_key(d["response"])
            if key_i in seen_keys or key_r in seen_keys:
                continue
            if args.diversify_func_names:
                code_block = find_code_blocks(d["response"])[0]
                try:
                    fn_names = extract_and_concat_function_names(code_block)
                except SyntaxError:
                    continue
                if fn_names in seen_keys:
                    continue
                seen_keys.add(fn_names)
            new_data.append(d)
            seen_keys.add(key_i)
            seen_keys.add(key_r)
        print("Non exact matches:", len(new_data))
    else:
        new_data = raw_data.to_list()
    new_dataset = Dataset.from_list(new_data)

    if args.data_augmentation:
        new_dataset = new_dataset.map(
            augment_data,
            num_proc=args.n_cores,
            with_indices=True,
        )
        print("Augmented:", len(new_dataset))

    write_jsonl(Path(args.output_file), new_dataset)


if __name__ == "__main__":
    main()
