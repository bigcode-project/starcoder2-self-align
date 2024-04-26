import ast
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

from tqdm.auto import tqdm
from transformers import HfArgumentParser

from star_align.utils import find_code_blocks, read_jsonl, write_jsonl


@dataclass(frozen=True)
class Args:
    data_files: list[str]
    output_file: str
    diversify_func_names: bool = field(default=False)


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


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    raw_data: list[dict] = []
    for data_file in args.data_files:
        data = read_jsonl(Path(data_file))
        # language = data_file.split("-")[1]
        # assert language in ALL_LANGS, f"Unknown language {language}"
        # raw_data.extend(dict(lang=language, **d) for d in data)
        raw_data.extend(data)
    # common keys for all d in data
    common_keys = set.intersection(*(set(d.keys()) for d in raw_data))
    raw_data = [{k: d[k] for k in common_keys} for d in raw_data]
    print(f"Common keys: {common_keys}")
    # counter = defaultdict[str, int](int)

    def mk_key(instruction: str) -> str:
        return "".join(instruction.split())

    random.seed(0)
    random.shuffle(raw_data)

    seen_keys = set[str]()
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

    print(f"Chose {len(new_data)} out of {len(raw_data)}")
    write_jsonl(Path(args.output_file), new_data)


if __name__ == "__main__":
    main()
