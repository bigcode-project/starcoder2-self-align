import ast
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Process, cpu_count
from evalplus.eval.utils import (
    create_tempdir,
    reliability_guard,
    swallow_io,
    time_limit,
)
from tqdm.auto import tqdm

from datasets import load_dataset
from star_align.utils import chunked, find_code_blocks
from transformers import HfArgumentParser
from dataclasses import dataclass, field
from typing import cast


_magic_splitter_ = "### -- what do you think? -- ###"


def make_python_membound_code_prefix(limit_mb):
    maximum_memory_bytes = limit_mb * 1024 * 1024
    return f"""\
import resource
import platform

resource.setrlimit(
    resource.RLIMIT_AS, ({maximum_memory_bytes}, {maximum_memory_bytes})
)
resource.setrlimit(
    resource.RLIMIT_DATA, ({maximum_memory_bytes}, {maximum_memory_bytes})
)
if not platform.uname().system == "Darwin":
    resource.setrlimit(
        resource.RLIMIT_STACK, ({maximum_memory_bytes}, {maximum_memory_bytes})
    )
{_magic_splitter_}
"""


@dataclass(frozen=True)
class Args:
    response_paths: list[str]
    result_path: str
    save_request_errors: bool = False
    shuffle: bool = field(default=True)
    cache_paths: list[str] = field(default_factory=list)
    load_pass_only_cache: bool = field(default=False)
    max_batched_tasks: int = 10000
    max_workers: int = cpu_count()
    container_server: str | None = None


def suppress_output(func):
    def wrapper(*args, **kwargs):
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        sys.stdout = open(os.devnull, "w")
        sys.stderr = sys.stdout
        try:
            result = func(*args, **kwargs)
        finally:
            sys.stdout.close()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
        return result

    return wrapper


# Note: only run this within a safe subprocess
def _run(code) -> None:
    with create_tempdir():
        # These system calls are needed when cleaning up tempdir.
        rmtree = shutil.rmtree
        rmdir = os.rmdir
        chdir = os.chdir
        getcwd = os.getcwd

        maximum_memory_bytes = 1 * 1024 * 1024 * 1024
        reliability_guard(maximum_memory_bytes=maximum_memory_bytes)

        # Disable functionalities that can make destructive changes to the test.
        # allow only 1GB memory usage

        # run the function
        with swallow_io():
            with time_limit(4):  # max 4 seconds
                # run the function
                exec(code)

        # Needed for cleaning up.
        shutil.rmtree = rmtree
        os.rmdir = rmdir
        os.chdir = chdir
        os.getcwd = getcwd


def containerized_run(item, limit_mb=4 * 1024):
    from star_align.code_exec_server.code_exec_reqs import exec_test

    idx, result, code, srv = item
    membound_code = make_python_membound_code_prefix(limit_mb) + code
    passed, output = exec_test(
        srv, membound_code, "", timeout=10, timeout_on_client=True
    )
    return (idx, result, code, passed, output)


def fork_run(item):
    idx, response, code, _ = item
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout
    p = Process(target=_run, args=(code,))
    p.start()
    p.join(timeout=10)
    passed = p.exitcode == 0
    return (idx, response, code, passed, "NOT SUPPORTED")


def is_compilable(code):
    try:
        ast.parse(code)
        return True
    except (SyntaxError, ValueError):
        return False


def extract_code(response: str) -> str:
    def sanitize_codeblock(code: str) -> str:
        if "input" not in code:
            return code.strip()
        # Only remove the `if __name__..` when `input` is present because
        # it will block the code execution.
        key = 'if __name__ == "__main__":'
        key_alt = "if __name__ == '__main__':"
        index = code.find(key)
        if index == -1:
            index = code.find(key_alt)
        if index == -1:
            return code.strip()
        assert index != -1
        code = code[:index].strip()
        return code

    code_blocks = list(map(sanitize_codeblock, find_code_blocks(response)))
    return "\n\n".join(code_blocks)


def form_new_data(
    item: dict,
    response: str,
    extracted_code: str,
    pass_execution: bool,
    output: str,
) -> dict:
    newdata = {k: v for k, v in item.items() if k not in ["response", "parsing_result"]}
    newdata["response"] = response
    newdata["extracted_code"] = extracted_code
    newdata["pass"] = pass_execution
    newdata["output"] = output
    return newdata


def main():
    args = cast(Args, HfArgumentParser(Args).parse_args_into_dataclasses()[0])
    if args.container_server is None:
        option = input(
            "WARNING: container_server is not set. You will run the code locally, which can lead to unexpected side effects. Continue? (y/n): "
        ).strip()
        if option.lower() != "y":
            return

    if os.path.exists(args.result_path):
        option = input(
            f"WARNING: {args.result_path} already exists. Overwrite? (y/n): "
        ).strip()
        if option.lower() != "y":
            return

    cleanup_command = os.getenv("CLEANUP_COMMAND", None)
    if cleanup_command is not None:
        print(f"NOTE: the cleanup command is set to:")
        print(cleanup_command)

    raw_data = load_dataset("json", data_files=args.response_paths, split="train")
    if args.shuffle:
        raw_data = raw_data.shuffle()
    if len(args.cache_paths) > 0:
        cached_data = load_dataset("json", data_files=args.cache_paths, split="train")
        if args.load_pass_only_cache:
            cached_dict: dict[str, dict] = {
                item["extracted_code"]: item for item in cached_data if item["pass"]
            }
        else:
            cached_dict = {item["extracted_code"]: item for item in cached_data}
    else:
        cached_dict = {}

    all_tasks: list[tuple[int, str, str, str | None]] = []
    eval_results: list[dict] = []
    for idx, item in enumerate(tqdm(raw_data, desc="Preprocessing: extracting code")):
        # passing_results = []
        if "parsing_result" not in item:
            item["parsing_result"] = [dict(response=item["response"])]
        for result in item["parsing_result"]:
            response = result["response"]
            code = extract_code(response)
            if (hit_item := cached_dict.get(code, None)) is not None:
                assert code == hit_item["extracted_code"]
                new_data = form_new_data(
                    item=item,
                    response=response,
                    extracted_code=code,
                    pass_execution=hit_item["pass"],
                    output=hit_item["output"],
                )
                eval_results.append(new_data)
            else:
                all_tasks.append((idx, response, code, args.container_server))

    def pass_rate_str(passed: int, total: int, tag: str = "") -> str:
        percentage = f"{passed/total * 100:.2f}%" if total > 0 else "N/A"
        ratio = f"{passed}/{total}"
        tag = f"{tag} " if len(tag) > 0 else ""
        return f"{tag}Passed: {ratio} ({percentage})"

    n_cached_passed = sum(item["pass"] for item in eval_results)
    n_cached_total = len(eval_results)

    print(f"Cached: {len(eval_results)}, Active: {len(all_tasks)}")
    print(pass_rate_str(n_cached_passed, n_cached_total, "Cached"))

    run_func = containerized_run if args.container_server else fork_run
    tasks_chunks = list(chunked(all_tasks, args.max_batched_tasks))
    n_processed = 0
    n_passed = 0
    with open(args.result_path, "w") as f:
        for cached_result in eval_results:
            f.write(json.dumps(cached_result) + "\n")
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            pbar = tqdm(tasks_chunks)
            for chunked_tasks in pbar:
                futures = [executor.submit(run_func, task) for task in chunked_tasks]
                # NOTE: futures do not return in the same order as before
                pbar_inner = tqdm(
                    as_completed(futures),
                    total=len(futures),
                    leave=False,
                )
                n_passed_inner = 0
                for n_processed_inner, future in enumerate(pbar_inner, start=1):
                    n_processed += 1
                    try:
                        future_result = future.result()
                    except Exception as e:
                        continue
                    idx, response, code, passed, output = future_result
                    if "Failed to execute program" in output:
                        if not args.save_request_errors:
                            continue
                    newdata = form_new_data(
                        item=raw_data[idx],
                        response=response,
                        extracted_code=code,
                        pass_execution=passed,
                        output=output,
                    )
                    f.write(json.dumps(newdata) + "\n")
                    n_passed += passed
                    n_passed_inner += passed
                    pbar_inner.set_description(
                        pass_rate_str(n_passed_inner, n_processed_inner)
                    )
                pbar.set_description(pass_rate_str(n_passed, n_processed))
                if cleanup_command is not None:
                    print(f"Cleaning up: {cleanup_command}")
                    os.system(cleanup_command)
                    print("Cleanup done.")

    n_total_passed = n_cached_passed + n_passed
    n_total = len(all_tasks) + n_cached_total
    print(pass_rate_str(n_total_passed, n_total, "Total"))


if __name__ == "__main__":
    main()
