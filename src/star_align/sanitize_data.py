"""Deduplication, filtering, and selection"""

import random
import os
import ast
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast, Literal
from datasets import load_dataset, Dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser

from star_align.utils import find_code_blocks, write_jsonl, find_codeblock_indices

LLAMA3 = os.getenv("LLAMA3") is not None
if LLAMA3:
    print("LLAMA3 mode activated")


@dataclass(frozen=True)
class Args:
    data_files: list[str]
    output_file: str
    shuffle: bool = field(default=True)
    remove_strange: bool = field(default=True)
    parse_raw_response: bool = field(default=True)
    passing_only: bool = field(default=True)
    data_augmentation: bool = field(default=False)
    exact_match_dedup: bool = field(default=True)
    get_code_representation: bool = field(default=True)
    remove_comments_docstrings: bool = field(default=False)
    include_left_failed: bool = field(default=False)
    n_cores: int = field(default=os.cpu_count() or 1)
    diversify_func_names: bool = field(default=True)
    align_with: list[str] = field(default_factory=list)
    priority: Literal["passed", "failed", "none"] = field(default="none")
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
    function_names = []
    class_names = []

    # Define a node visitor that adds the name of each function definition it visits
    class FuncClassDefVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node: ast.ClassDef):
            class_names.append(node.name)
            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            function_names.append(node.name)
            # Process the subtree for this node
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            function_names.append(node.name)
            self.generic_visit(node)

    # Create a node visitor and walk through the AST
    visitor = FuncClassDefVisitor()
    visitor.visit(tree)

    def compress_name(name: str) -> str:
        return name.replace("_", "").lower()

    return frozenset(map(compress_name, function_names)), frozenset(
        map(compress_name, class_names)
    )


INCOMPLETE_SUBSTRINGS = [
    "todo",
    "fixme",
    "write your code here",
    "your code here",
    "your code goes here",
    "notimplemented",
]

RESPONSE_TEST_SPLIT = "</response>\n\n<tests>"
# special handling for llama3 since it has more examples not following the format
LLAMA3_DEFAULT_TEST_SPLIT = r"### Tests \d\n"
LLAMA3_ADDITIONAL_PATTERNS = [
    "We can verify the functionality",
    "We can verify the correctness",
    "You can verify the correctness",
    "You can verify the functionality",
    "To ensure the correctness",
    "To verify the correctness",
    "To test the",
    "To test this",
    "To test this",
    "You can test the",
    "We can test the",
    "We can test this",
    "Now, we'll test",
]


def split_llama3_response_tests(response: str) -> list[str]:
    splits = re.split(LLAMA3_DEFAULT_TEST_SPLIT, response)
    if len(splits) > 2:
        return []
    if len(splits) == 2:
        return splits
    for pattern in LLAMA3_ADDITIONAL_PATTERNS:
        index = response.find(pattern)
        if index != -1:
            return [response[:index], response[index:]]
    return []


def preprocess_and_filter(x: dict) -> dict:
    """Filter out responses with wrong format"""

    def wrong_format(x: dict) -> dict:
        return {k: v for k, v in x.items()} | dict(wrong_format=True, tests="<NO>")

    response: str = x["response"]
    if not LLAMA3 and RESPONSE_TEST_SPLIT not in response:
        return wrong_format(x)
    if any(substring in response.lower() for substring in INCOMPLETE_SUBSTRINGS):
        return wrong_format(x)
    if LLAMA3:
        splits = split_llama3_response_tests(response)
    else:
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
        response=response, tests=tests, wrong_format=False
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


def remove_ast_docstrings(tree):
    # ref: https://gist.github.com/phpdude/1ae6f19de213d66286c8183e9e3b9ec1
    for node in ast.walk(tree):
        # let's work only on functions & classes definitions
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            continue
        if len(node.body) == 0:
            continue
        if not isinstance(node.body[0], ast.Expr):
            continue
        if (
            not hasattr(node.body[0], "value")
            or not isinstance(node.body[0].value, ast.Str)
            # or not isinstance(node.body[0].value.value, str)
        ):
            continue
        node.body = node.body[1:]  # type: ignore
    return tree


def remove_comments_from_code_blocks(
    content: str,
) -> str:
    code_blocks = find_codeblock_indices(content)
    # Current index in the original content for tracking purposes
    current_index = 0
    # Buffer to store the new content
    new_content: list[str] = []
    # Iterate over each code block
    for start, end in code_blocks:
        # Append the content before this code block
        new_content.append(content[current_index:start])

        # Extract the code block content
        code_block_content = content[start:end]

        # Split into lines, process, and rejoin
        modified_block_content = remove_comments(code_block_content)

        new_content.append(modified_block_content)

        # Update current index
        current_index = end

    # Add the remaining part of the original content after the last code block
    new_content.append(content[current_index:])

    # Join all parts to form the final modified content
    return "".join(new_content)


def remove_comments(code: str) -> str:
    """Remove comments and docstrings using AST"""
    tree = ast.parse(code)
    tree = remove_ast_docstrings(tree)
    return ast.unparse(tree)


def get_code_representation(response: str) -> str:
    """Keep classes and functions, removing comments and docstrings"""
    raw_code = "\n".join(find_code_blocks(response))

    tree = ast.parse(raw_code)

    class ClassFunctionTransformer(ast.NodeTransformer):
        def visit_Module(self, node):
            # Visit all children nodes of the module
            node = self.generic_visit(node)
            # Filter out only function and class definitions
            node.body = [
                n for n in node.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            ]
            return node

    visitor = ClassFunctionTransformer()
    tree = visitor.visit(tree)
    tree = remove_ast_docstrings(tree)
    return ast.unparse(tree)


def map_code_representation(x: dict) -> dict:
    try:
        representation = get_code_representation(x["response"])
    except SyntaxError:
        representation = "<SYNTAX ERROR>"
    return {k: v for k, v in x.items()} | dict(code_representation=representation)


# def concat_list(lists: list[list]) -> list:
#     return [item for sublist in lists for item in sublist]


def map_examples_batched(examples: dict, map_one) -> dict:
    all_keys = list(examples.keys())
    list_of_examples = [
        {k: examples[k][i] for k in all_keys} for i in range(len(examples[all_keys[0]]))
    ]
    results = [map_one(example) for example in list_of_examples]
    result_dict = {k: [result[k] for result in results] for k in results[0].keys()}
    return result_dict


def map_remove_comments(x: dict) -> dict:
    try:
        response = x["response"]
    except SyntaxError:
        response = "<SYNTAX ERROR>"
    return {k: v for k, v in x.items() if k != "response"} | dict(response=response)


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])

    raw_data = load_dataset("json", data_files=args.data_files, split="train")
    if args.align_with:
        ref_data = load_dataset("json", data_files=args.align_with, split="train")
        ref_data_instructions = set(map(lambda x: x["instruction"], ref_data))
        raw_data = raw_data.filter(
            lambda x: x["instruction"] in ref_data_instructions, num_proc=args.n_cores
        )
    print("Raw samples:", len(raw_data))

    if args.parse_raw_response:
        raw_data = raw_data.map(
            map_examples_batched,
            fn_kwargs=dict(map_one=preprocess_and_filter),
            batched=True,
            num_proc=args.n_cores,
        )
        raw_data = raw_data.filter(
            lambda x: not x["wrong_format"], num_proc=args.n_cores
        )
        raw_data = raw_data.remove_columns(["wrong_format"])
        print("Correct format:", len(raw_data))

    if args.include_left_failed:
        failed_data = raw_data.filter(lambda x: not x["pass"], num_proc=args.n_cores)

    if args.passing_only:
        raw_data = raw_data.filter(lambda x: x["pass"], num_proc=args.n_cores)
        print("Passing only:", len(raw_data))

    if args.shuffle:
        raw_data = raw_data.shuffle(seed=args.seed)
        if args.include_left_failed:
            failed_data = failed_data.shuffle(seed=args.seed)

    if args.priority != "none":
        # Sort the examples such that failed/passed are at first
        raw_data = raw_data.map(
            map_examples_batched,
            fn_kwargs=dict(map_one=lambda x: dict(**x, rank=int(x["pass"]))),
            batched=True,
            num_proc=args.n_cores,
        )
        reverse = args.priority == "passed"
        raw_data = raw_data.sort(column_names="rank", reverse=reverse)
        raw_data = raw_data.remove_columns("rank")

    def mk_key(instruction: str) -> str:
        return "".join(instruction.split())

    seen_ids = set[frozenset[str]]()
    seen_keys = set[str]()
    if args.exact_match_dedup:
        new_data = list[dict]()

        def iterate(dataset: Dataset):
            for d in tqdm(dataset):
                if args.remove_strange:
                    # NOTE: newly added
                    if len(d["instruction"].split()) > 200:
                        continue
                key_i, key_r = mk_key(d["instruction"]), mk_key(d["response"])
                if key_i in seen_keys or key_r in seen_keys:
                    continue
                if args.diversify_func_names:
                    code_block = find_code_blocks(d["response"])[0]
                    try:
                        fn_names, class_names = extract_and_concat_function_names(
                            code_block
                        )
                    except SyntaxError:
                        continue
                    if (len(fn_names) > 0 and fn_names in seen_ids) or (
                        len(class_names) > 0 and class_names in seen_ids
                    ):
                        continue
                    seen_ids.add(fn_names)
                    seen_ids.add(class_names)
                new_data.append(d)
                seen_keys.add(key_i)
                seen_keys.add(key_r)

        iterate(raw_data)
        if args.include_left_failed:
            iterate(failed_data)

        print("Non exact matches:", len(new_data))
    else:
        new_data = raw_data.to_list()
        if args.include_left_failed:
            new_data.extend(failed_data.to_list())
    new_dataset = Dataset.from_list(new_data)

    if args.get_code_representation:
        new_dataset = new_dataset.map(
            map_examples_batched,
            fn_kwargs=dict(map_one=map_code_representation),
            batched=True,
            batch_size=1000,
            # num_proc=args.n_cores,
        )
        new_dataset = new_dataset.filter(
            lambda x: x["code_representation"] != "<SYNTAX ERROR>",
            num_proc=args.n_cores,
        )
        print("Extracted code representation:", len(new_dataset))

    if args.remove_comments_docstrings:
        new_dataset = new_dataset.map(
            map_examples_batched,
            fn_kwargs=dict(map_one=map_remove_comments),
            batched=True,
            # num_proc=args.n_cores,
        )
        new_dataset = new_dataset.filter(
            lambda x: x["response"] != "<SYNTAX ERROR>",
            num_proc=args.n_cores,
        )
        print("Removed comments/docstrings:", len(new_dataset))

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
