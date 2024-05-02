import ast
import json
import os
import re
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
from fire import Fire
from tqdm.auto import tqdm

from star_align.utils import chunked

# dependencies: evalplus fire tqdm


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


# from code_exec_server.code_exec_reqs import exec_test
def containerized_run(item):
    from code_exec_server.code_exec_reqs import exec_test

    idx, result, code, srv = item
    passed, _ = exec_test(srv, code, "", timeout=10)
    return (idx, result) if passed else None


def fork_run(item):
    idx, result, code, _ = item
    sys.stdout = open(os.devnull, "w")
    sys.stderr = sys.stdout
    p = Process(target=_run, args=(code,))
    p.start()
    p.join(timeout=10)
    return (idx, result) if p.exitcode == 0 else None


def is_compilable(code):
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def extract_code(response):
    pattern = r"^```python\s*\n(.*?)(?=^```)"
    result = re.findall(pattern, response, re.DOTALL | re.MULTILINE)
    return "\n".join([x for x in result if is_compilable(x)])


# /scratch/sc2-instruct/data-ossx-del2-fewshot-mpt-response-temp0-s_i-1shot-temp0-i_r-8754b-0-20240321_172151.jsonl
def main(
    response_path: str,
    result_path: str,
    max_workers: int = cpu_count(),
    cache_path: str | None = None,
    container_server=None,
):
    # load jsonl
    with open(response_path, "r") as f:
        raw_data = [json.loads(line) for line in f if line.strip()]
    if cache_path is not None:
        with open(cache_path, "r") as f:
            cached_data = [json.loads(line) for line in f if line.strip()]
        # instruction -> set[response]
        hit_code = set[str]()
        for item in tqdm(cached_data):
            code = extract_code(item["response"])
            hit_code.add(code)

    uncompilable = 0
    all_tasks = []

    print("Container server:", container_server)

    for idx, item in enumerate(tqdm(raw_data)):
        # passing_results = []
        if "parsing_result" not in item:
            code = extract_code(item["response"])
            if not code:
                uncompilable += 1
                continue
            all_tasks.append((idx, item, code, container_server))
        else:
            for result in item["parsing_result"]:
                code = extract_code(result["response"])
                if not code:
                    uncompilable += 1
                    continue
                all_tasks.append((idx, result, code, container_server))
                # passing_results.append((result, code))

    # Split cached/un-cached data
    active_tasks = []
    cached_tasks = []
    for task in tqdm(all_tasks):
        _, _, code, _ = task
        if cache_path is not None and code in hit_code:
            cached_tasks.append(task)
        else:
            active_tasks.append(task)

    with open(result_path, "w") as f:
        for idx, result, _, _ in cached_tasks:
            newdata = {
                k: v
                for k, v in raw_data[idx].items()
                if k not in ["response", "parsing_result"]
            }
            newdata["response"] = result["response"]
            f.write(json.dumps(newdata) + "\n")

    print(f"Active tasks: {len(active_tasks)}")
    print(f"Cached tasks: {len(cached_tasks)}")

    run_func = containerized_run if container_server else fork_run

    nfails = 0
    tasks_chunks = chunked(active_tasks, os.cpu_count())
    with open(result_path, "a") as f:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for chunked_tasks in tqdm(tasks_chunks):
                futures = [executor.submit(run_func, task) for task in chunked_tasks]
                # for idx, presults in tqdm(tasks):
                #     futures = [
                #         executor.submit(fork_run, (i, pres))
                #         for i, pres in enumerate(presults)
                #     ]
                #     passed_indices = []
                # NOTE: futures do not return in the same order as before
                for future in tqdm(as_completed(futures), total=len(futures)):
                    try:
                        future_result = future.result()
                        if future_result is None:
                            nfails += 1
                            continue
                        idx, result = future_result
                        newdata = {
                            k: v
                            for k, v in raw_data[idx].items()
                            if k not in ["response", "parsing_result"]
                        }
                        newdata["response"] = result["response"]
                        f.write(json.dumps(newdata) + "\n")
                    except Exception:
                        nfails += 1
                        continue
                # if passed_indices:
                #     item = data[idx]
                #     item["parsing_result"] = [presults[i] for i in passed_indices]
                #     f.write(json.dumps(item) + "\n")

    print(f"Uncompilable: {uncompilable}")
    print(f"Failed: {nfails}")


if __name__ == "__main__":
    print("Try to run this file using docker if possible!")
    Fire(main)
