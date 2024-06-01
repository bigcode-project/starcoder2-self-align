import json
import random
import re
import textwrap
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from tqdm.auto import tqdm
from transformers import HfArgumentParser
import star_align

print("Warning: ignoring warnings")
warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class Args:
    data_dirs: list[str]
    data_mix_weights: list[float]

    max_seeds_to_collect: int = field(default=100000000)
    continue_from: str | None = field(default=None)

    # Keep the following arguments unchanged for reproducibility
    seed: int = field(default=976)

    min_lines: int = field(default=5)
    max_lines: int = field(default=30)
    min_doc_lines: int = field(default=10)
    max_doc_lines: int = field(default=5000)
    max_avg_chars_per_line: int = field(default=80)
    # max_fragments: int = field(default=3)
    chunk_size: int = field(default=1000)
    # A small value lets one document be used by multiple seeds
    content_chunk_lines: int = field(default=100)

    dataset_name: str = field(default="bigcode/starcoderdata")
    data_files: list[str] | None = field(default=None)
    max_considered_data: int | None = field(default=500000000)

    collect_function: bool = field(default=False)
    max_nodes_to_traverse: int = field(default=20000)

    tag: str = field(
        default="",
        metadata={
            "help": "Custom tag as part of the output filename, not affecting the fingerprint"
        },
    )
    text_wrap: int | None = field(default=None)

    std_lib_only: bool = field(default=False)
    min_stars: int = field(default=0)
    n_cores: int = field(default=star_align.utils.N_CORES)

    def fingerprint(self) -> str:
        # The combination of arguments can uniquely determine the generation process
        args = [
            self.data_dirs,
            self.data_mix_weights,
            self.seed,
            self.min_lines,
            self.max_lines,
            self.min_doc_lines,
            self.max_doc_lines,
            self.max_avg_chars_per_line,
            self.chunk_size,
            self.dataset_name,
            self.max_considered_data,
            self.content_chunk_lines,
            self.min_stars,
            self.text_wrap,
            self.data_files,
            self.collect_function,
            self.max_nodes_to_traverse,
        ]
        # for backward compatibility, only add if needed
        # if self.text_wrap is not None:
        #     args.append(self.text_wrap)
        # if self.data_files is not None:
        #     args.append(self.data_files)
        return star_align.utils.compute_fingerprint(*args, hash_length=5)


# def fragments_to_text(fragments: list[str]) -> str:
#     return "...\n".join(fragments)


ASTTypeMap = {
    "cpp": [
        "function_definition"
    ],  # https://github.com/tree-sitter/tree-sitter-cpp/blob/master/grammar.js
    # "csharp": [
    #     "method_declaration"
    # ],  # https://github.com/tree-sitter/tree-sitter-c-sharp/blob/master/grammar.js
    "java": [
        "method_declaration"
    ],  # https://github.com/tree-sitter/tree-sitter-java/blob/master/grammar.js
    "php": [
        "method_declaration"
    ],  # https://github.com/tree-sitter/tree-sitter-php/blob/master/grammar.js
    "python": [
        "function_definition"
    ],  # https://github.com/tree-sitter/tree-sitter-python/blob/master/grammar.js
    "rust": [
        "function_item"
    ],  # https://github.com/tree-sitter/tree-sitter-rust/blob/master/grammar.js
    # "swift": [
    #     "function_declaration"
    # ],  # https://github.com/alex-pinkus/tree-sitter-swift/blob/main/grammar.js
    "typescript": [
        "function_declaration",
        "method_definition",
    ],  # https://github.com/tree-sitter/tree-sitter-typescript/blob/master/typescript/grammar.js
    # "bash": [
    #     "function_definition"
    # ],  # https://github.com/tree-sitter/tree-sitter-bash/blob/master/grammar.js
}

EXCLUDED_PATTERNS = {
    "python": {
        "function_definition": [
            "def __[a-zA-Z0-9_]*__\\(",  # Broadly exclude magic methods, but note this may also exclude some user-defined dunder methods which might be 'interesting'
            "def test[a-zA-Z0-9_]*\\(",  # Exclude test functions
            # ends with pass
            # "\\(self[,\\)]",
            "pass$",
        ]
    },
    "cpp": {
        "function_definition": [
            "~[a-zA-Z0-9_]*\\(",  # Exclude C++ destructors
        ]
    },
    "java": {
        "function_definition": [
            "[gs]et[A-Z][0-9a-zA-Z_]*\\(",
        ]
    },
    "php": {
        "function_definition": [
            "function __[A-Za-z0-9_]*",  # Exclude PHP magic methods
            "function test[a-zA-Z0-9_]*\\(",  # Exclude PHP magic methods
        ]
    },
    "rust": {
        "function_definition": [
            "fn new(",
            "fn default(",
            "fn test[a-zA-Z0-9_]*\\(",
        ]
    },
    "typescript": {
        "function_definition": [
            "constructor(",
        ]
    },
}


def extract_snippets_with_constraints(
    tree,
    source_code: str,
    considered_types: list[str],
    min_lines: int,
    max_lines: int,
    max_nodes_to_traverse: int,
    excluded_patterns: dict[str, list[str]],
) -> list[str]:
    """
    Extract snippets from the source code that match the given constraints.
    """
    from tree_sitter import Node, Tree
    assert isinstance(tree, Tree)

    matching_snippets = list[str]()
    # source_code = root_node.text.decode()

    # https://github.com/tree-sitter/py-tree-sitter/blob/master/examples/walk_tree.py
    def traverse_tree(tree: Tree) -> Generator[Node, None, None]:
        cursor = tree.walk()

        visited_children = False
        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

    all_nodes_iter = traverse_tree(tree)
    for node, _ in zip(all_nodes_iter, range(max_nodes_to_traverse)):
        if node.type in considered_types:
            start_line = node.start_point[0]  # line numbers are zero-indexed
            end_line = node.end_point[0]
            line_count = end_line - start_line + 1

            if min_lines <= line_count <= max_lines:
                snippet = extract_code_with_indentation(node, source_code)
                # Check if the fragment contains any of the excluded keywords
                if all(
                    re.search(pattern, snippet) is None
                    for pattern in excluded_patterns.get(node.type, [])
                ):
                    snippet = textwrap.dedent(snippet)
                    matching_snippets.append(snippet)

    return matching_snippets


def extract_code_with_indentation(node, source_code: str) -> str:
    """
    Extract the source code corresponding to a given node from the original source code string,
    including the indentation based on the node's starting line.
    """
    start_byte = node.start_byte
    end_byte = node.end_byte
    # Find the newline character before the node starts, to determine the start of the line
    start_of_line = (
        source_code.rfind("\n", 0, start_byte) + 1
    )  # +1 to move past the newline character
    # Calculate the indentation by counting whitespace characters from the start of the line to the node start
    indentation = ""
    for i in range(start_of_line, start_byte):
        if source_code[i].isspace():
            indentation += source_code[i]
        else:
            break
    # Extract the code and prepend the indentation to each line
    code_fragment = source_code[start_byte:end_byte]
    indented_code_fragment = indentation + code_fragment
    return indented_code_fragment


def chunk_content(examples: dict, indices: list[int], args: Args) -> dict:
    contents = examples["content"]
    examples["raw_index"] = indices

    def chunk(content: str) -> list[str]:
        lines = content.splitlines(keepends=True)
        chunks = list[str]()
        for end_idx in range(len(lines), 0, -args.content_chunk_lines):
            # throw away the last chunk if it's too small
            if end_idx < args.content_chunk_lines and len(chunks) > 0:
                break
            chunks.append("".join(lines[end_idx - args.content_chunk_lines : end_idx]))
        return chunks

    new_data: dict = dict()
    for index in range(len(contents)):
        content = contents[index]
        chunked_contents = chunk(content)
        new_data.setdefault("chunked_content", []).extend(chunked_contents)
        for key in examples.keys():
            new_others = [examples[key][index]] * len(chunked_contents)
            new_data.setdefault(key, []).extend(new_others)

    return new_data


def map_dataset(examples: dict, indices: list[int], args: Args, data_dir: str) -> dict:
    random.seed(args.seed + sum(map(ord, data_dir)) + indices[0])
    stars = list(map(int, examples["max_stars_count"]))
    content_key = "chunked_content" if not args.collect_function else "content"
    contents = list(map(sanitize_document, examples[content_key]))
    assert len(contents) == len(stars)
    if not args.collect_function:
        seed_fragments = [
            (
                extract_fragment(args, content)
                if content is not None and star >= args.min_stars
                else None
            )
            for star, content in zip(stars, contents)
        ]
        # seed = [
        #     (fragments_to_text(fragments) if fragments is not None else None)
        #     for fragments in seed_fragments
        # ]
        assert len(seed_fragments) == len(indices)
        return {
            "seed": seed_fragments,
            "repo": examples["max_stars_repo_name"],
            "star": stars,
            "id": list(map(int, indices)),
            "raw_index": examples["raw_index"],
            "data_dir": [data_dir for _ in indices],
            # "content": examples["content"]
        }
    from tree_sitter_languages import get_parser

    parser = get_parser(data_dir)
    repos = examples["max_stars_repo_name"]
    ids = examples["id"]
    raw_indices = indices
    data_dirs = [data_dir for _ in indices]
    data: dict[str, list] = {
        "seed": [],
        "repo": [],
        "star": [],
        "id": [],
        "raw_index": [],
        "data_dir": [],
    }
    for repo, star, id, raw_index, content, data_dir in zip(
        repos, stars, ids, raw_indices, contents, data_dirs
    ):
        if (
            content is None
            or star < args.min_stars
            or (n_lines := content.count("\n")) < args.min_doc_lines
            or n_lines > args.max_doc_lines
            or len(content) > args.max_avg_chars_per_line * args.max_doc_lines
        ):
            continue
        try:
            content_encoded = content.encode()
            if len(content_encoded) != len(content):
                # probably Non-english
                continue
            tree = parser.parse(content_encoded)
            root_node = tree.root_node
            if root_node.has_error:
                continue
            # if len(root_node.text) != len(content):
            #     # probably Non-english
            #     continue
            snippets = extract_snippets_with_constraints(
                tree=tree,
                source_code=content,
                considered_types=ASTTypeMap[data_dir],
                min_lines=args.min_lines,
                max_lines=args.max_lines,
                max_nodes_to_traverse=args.max_nodes_to_traverse,
                excluded_patterns=EXCLUDED_PATTERNS[data_dir],
            )
            data["seed"].extend(snippets)
            for key in ["repo", "star", "id", "raw_index", "data_dir"]:
                data[key].extend([locals()[key]] * len(snippets))
        except UnicodeError:
            pass
    return data


# def uniform_partition(n: int, k: int) -> list[int]:
#     """Partition n into k non-negative integers. Stars and bars method.
#     x1 + x2 + ... + xk = n; xi >= 0. Can be transformed to positive case:
#     y1 + y2 + ... + yk = n - k; yi = xi + 1 > 0"""
#     assert n >= 0, "n should be >= 0"
#     assert k > 0, " should be > 0"
#     random_numbers = [random.randint(0, n) for _ in range(k - 1)]
#     values = [0] + sorted(random_numbers) + [n]
#     intervals = [values[i + 1] - values[i] for i in range(len(values) - 1)]
#     assert sum(intervals) == n
#     assert len(intervals) == k
#     return intervals


# def uniform_partition_positive(n: int, k: int) -> list[int]:
#     return [x + 1 for x in uniform_partition(n - k, k)]


# def is_en(content: str, seed: int) -> bool:
#     import langdetect

#     langdetect.DetectorFactory.seed = seed
#     try:
#         return langdetect.detect(content) == "en"
#     except langdetect.lang_detect_exception.LangDetectException:
#         return False


# def place_blocks(n: int, sizes: list[int]) -> list[int]:
#     """Randomly place k blocks of sizes `sizes` in a line of length n. Return the starting positions."""
#     assert n >= 0, "n should be >= 0"
#     k = len(sizes)
#     assert k > 0, "k should be > 0"
#     assert sum(sizes) <= n, "sum(sizes) should be <= n"
#     if k == 1:
#         return [random.randint(0, n - sizes[0])]
#     all_but_last_pos = place_blocks(n - sizes[-1], sizes[:-1])
#     last_pos = random.randint(all_but_last_pos[-1] + sizes[-2], n - sizes[-1])
#     result = all_but_last_pos + [last_pos]
#     assert len(result) == k
#     for i in range(k - 1):
#         assert result[i] + sizes[i] <= result[i + 1]
#     return result


def sanitize_document(document: str) -> str | None:
    """Sanitize the document by removing the first line if it's a placeholder."""
    if (
        document.startswith("<reponame>")
        or document.startswith("<filename>")
        or document.startswith("<gh_stars>")
    ):
        # remove the first line
        newline_index = document.find("\n")
        if newline_index == -1:
            return None
        document = document[newline_index + 1 :]
    return document


def extract_fragment(args: Args, document: str) -> str | None:
    if args.std_lib_only:
        if not check_std_libs_only(document):
            return None
        else:
            return document
    if args.text_wrap is not None:
        document = textwrap.fill(
            document,
            width=args.text_wrap,
            replace_whitespace=False,
            expand_tabs=False,
            break_on_hyphens=False,
            drop_whitespace=False,
            break_long_words=False,
        )
    # if args.data_dir == "markdown" and not is_en(document, args.seed):
    #     return None
    document = document.replace("\r", "")
    document = re.sub(r"\n\n+", "\n\n", document)
    lines = document.splitlines(keepends=True)

    # special rules
    # if args.data_dir == "jupyter-scripts-dedup-filtered":
    #     lines = [
    #         line
    #         for line in lines
    #         if "jupyter" not in line.lower() and "jupytext" not in line.lower()
    #     ]
    # elif args.data_dir == "markdown":
    #     lines = [
    #         line
    #         for line in lines
    #         if "http:" not in line and "https:" not in line and "www." not in line
    #     ]

    # lines = [line for line in lines if line.strip() != ""]

    # if len(lines) < args.min_lines or len(lines) == 0:
    if len(lines) < args.min_doc_lines or len(lines) > args.max_doc_lines:
        return None
    # avg chars
    if len(document) > args.max_avg_chars_per_line * args.max_doc_lines:
        return None
    max_lines = min(args.max_lines, len(lines))
    assert args.max_lines >= args.min_lines
    n_lines_to_consider = random.randint(args.min_lines, max_lines)
    # max_fragments = min(n_lines_to_consider, args.max_fragments)
    # n_fragments = random.randint(1, max_fragments)
    # fragment_sizes = uniform_partition_positive(n_lines_to_consider, n_fragments)
    # fragment_indices = place_blocks(len(lines), fragment_sizes)
    # fragments = [
    #     "".join(lines[i : i + size])
    #     for i, size in zip(fragment_indices, fragment_sizes)
    # ]
    start_index = random.randint(0, len(lines) - n_lines_to_consider)
    # random.shuffle(fragments)
    content = "".join(lines[start_index : start_index + n_lines_to_consider])
    content = textwrap.dedent(content.replace("\t", "    "))
    return content


import sys


def is_std_lib(name):
    """Check if a module is a standard library."""
    return name in {*sys.builtin_module_names, *sys.stdlib_module_names}


def check_std_libs_only(code):
    """Check if all imported libraries in the given code are standard libraries."""
    lines = code.split("\n")
    for line in lines:
        if line.startswith("import ") or line.startswith("from "):
            parts = line.split()
            if len(parts) < 2:
                # special case
                return False
            if line.startswith("import "):
                module_name = parts[1].split(".")[0]  # Get the base module name
            else:  # from ... import ...
                module_name = parts[1]

            if not is_std_lib(module_name):
                return False
    return True


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    assert len(args.data_dirs) == len(args.data_mix_weights)
    sum_weights = sum(args.data_mix_weights)
    data_mix_ratios = [w / sum_weights for w in args.data_mix_weights]
    random.seed(args.seed)
    raw_datasets: list[Dataset] = []
    num_proc = args.n_cores
    # num_proc = 1
    for data_dir, ratio in zip(args.data_dirs, data_mix_ratios):
        max_considered_data = (
            None
            if args.max_considered_data is None
            else int(args.max_considered_data * ratio)
        )
        print(f"Loading {data_dir} with max_considered_data={max_considered_data}")
        split = (
            f"train[:{max_considered_data}]"
            if max_considered_data is not None
            else "train"
        )
        try:
            kwargs = dict(
                data_dir=data_dir,
                split=split,
                data_files=args.data_files,
                num_proc=num_proc,
                ignore_verifications=True,
            )
            sub_dataset = load_dataset(args.dataset_name, **kwargs)
        except ValueError:
            print(
                f"Failed to load {data_dir} with split=train[:{max_considered_data}]. Trying split=train"
            )
            kwargs["split"] = "train"
            sub_dataset = load_dataset(args.dataset_name, **kwargs)
        if not args.collect_function:
            sub_dataset = sub_dataset.map(
                function=chunk_content,
                fn_kwargs=dict(args=args),
                with_indices=True,
                batched=True,
                batch_size=args.chunk_size,
                num_proc=num_proc,
                remove_columns=["content"],
            )
        raw_datasets.append(sub_dataset)
    # map_fn = get_map_dataset(args)
    datasets: list[Dataset] = []
    for data_dir, sub_dataset in zip(args.data_dirs, raw_datasets):
        sub_dataset = sub_dataset.map(
            function=map_dataset,
            fn_kwargs=dict(args=args, data_dir=data_dir),
            with_indices=True,
            batched=True,
            num_proc=num_proc,
            batch_size=args.chunk_size,
            remove_columns=sub_dataset.column_names,
            load_from_cache_file=False,
        )
        datasets.append(sub_dataset)
    dataset = concatenate_datasets(datasets)
    dataset = dataset.shuffle(seed=args.seed)

    # Every run should produce the same data as long as the default params are not changed
    data_fingerprint = args.fingerprint()
    timestamp = star_align.utils.timestamp()
    tag = "" if args.tag == "" else f"-{args.tag}"
    path = Path(f"data-seed{tag}-{data_fingerprint}-{timestamp}.jsonl")
    assert not path.exists()
    with path.open("w") as f_out:
        print("Saving to", path)

        n_success = 0
        all_seed_texts = set[str]()

        def get_seed_text(seed: str) -> str:
            return "".join(seed.split())

        pbar = tqdm(total=min(args.max_seeds_to_collect, len(dataset)))
        for example in dataset:
            if n_success >= args.max_seeds_to_collect:
                break
            if example["seed"] is None:
                continue
            seed_text = get_seed_text(example["seed"])
            # remove those with only symbols
            if all(not c.isalpha() for c in seed_text):
                # print("[filter(symbols Only)]", example["seed"], sep="\n")
                continue
            if seed_text in all_seed_texts:
                # print("[filter(duplicate)]", example["seed"], sep="\n")
                continue
            all_seed_texts.add(seed_text)
            data = example
            # print("[Seed]", example["seed"], sep="\n")
            f_out.write(json.dumps(data) + "\n")
            n_success += 1
            pbar.update(1)
        print("Success:", n_success)


if __name__ == "__main__":
    main()
