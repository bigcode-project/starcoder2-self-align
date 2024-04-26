import datasets
import subprocess
import tempfile
import signal
import hashlib
import os
import argparse
from typing import List, Dict
from tqdm import tqdm

from tree_sitter_parser import LANGUAGE, global_parser

RETURN_QUERY = LANGUAGE.query("""
(return_statement) @return
""")


def does_have_return(src):
    tree = global_parser.parse(bytes(src, "utf8"))
    root = tree.root_node
    captures = RETURN_QUERY.captures(root)
    for node, _ in captures:
        # if it doesn't have an argument, it's not a return with a value
        if len(node.children) <= 1:  # includes "return" itself
            continue
        else:
            return True

    return False


if __name__ == "__main__":
    print(does_have_return("def foo():\n    return"))


# runs pyright in the given directory, returns stdout
# then, it logs the number of errors for each file
def run_pyright(d):
    try:
        outs = subprocess.run(
            ["pyright", "*"],
            cwd=d,
            capture_output=True,
            timeout=120,
            text=True,
        ).stdout
    except Exception as e:
        print(e)
        return None

    cur_file = ""
    filemap = {}
    lines = outs.split("\n")
    for i, line in enumerate(lines):
        if i == len(lines) - 2:
            break

        if line.startswith("  "):
            if "- error:" in line:
                filemap[cur_file] += 1
        else:
            file = line.split("/")[-1]
            filemap[file] = 0
            cur_file = file

    return filemap


def typecheck_batch(files: List[str]) -> Dict[str, str]:
    # Create a temporary directory using the tempfile module
    filemap: Dict[str, str] = {}
    with tempfile.TemporaryDirectory() as tempdir:
        for contents in files:
            hash_object = hashlib.sha1(bytes(contents, "utf8"))
            hex_dig = hash_object.hexdigest()
            filemap[hex_dig] = contents
            name = os.path.join(tempdir, hex_dig + ".py")
            with open(name, "w") as f:
                f.write(contents)

        # Run pyright in the temporary directory
        typecheck_map = run_pyright(tempdir)
        if typecheck_map is None:
            return {}

    for contents, errors in typecheck_map.items():
        no_py = contents.replace(".py", "")
        if errors == 0:
            continue

        if no_py in filemap:
            del filemap[no_py]

    print(f"Pass rate: {len(filemap)}/{len(files)}")

    return filemap


def infer_imports(code: str) -> str:
    import autoimport

    try:
        def handler(signum, frame):
            raise Exception("Timeout")
        signal.signal(signal.SIGALRM, handler)
        signal.alarm(10)
        inferred = autoimport.fix_code(code)
        signal.alarm(0)
        return inferred
    except Exception as e:
        signal.alarm(0)
        print(f"Error while inferring imports: {e}")
        return code


def main(args):
    ds = datasets.load_dataset(args.dataset,
                               data_dir="data", split="train")

    print("Filtering to only functions with return statements")
    ds = ds.filter(lambda ex: does_have_return(
        ex["content"]), num_proc=os.cpu_count())

    if args.infer_imports:
        print("Inferring imports for functions")
        ds = ds.map(lambda ex: {"content": infer_imports(
            ex["content"])}, num_proc=os.cpu_count())

    batch = []
    max_i = len(ds) - 1

    new_ds = {
        "content": [],
        "sha1": [],
        "id": [],
    }

    e_id = 0

    for i, ex in enumerate(tqdm(ds, total=len(ds))):
        try:
            code = ex["content"]

            batch.append(code)

            if len(batch) == args.batch_size or i == max_i:
                filemap = typecheck_batch(batch)
                for sha1, contents in filemap.items():
                    new_ds["content"].append(contents)
                    new_ds["sha1"].append(sha1)
                    new_ds["id"].append(e_id)
                    e_id += 1

                batch = []
        except Exception as e:
            print(f"There was an error: {e}")
            continue

    new_ds_hf = datasets.Dataset.from_dict(new_ds)
    new_ds_hf.push_to_hub(args.push, private=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        help="Points to dataset of python functions with docstrings. Columns: 'content'",
                        required=True)
    parser.add_argument(
        "--push", type=str, required=True, help="Push to this dataset to which repo")
    parser.add_argument(
        "--infer-imports", action="store_true", help="Infer imports for functions")
    parser.add_argument(
        "--batch-size", type=int, default=250, help="Batch size for typechecking")
    args = parser.parse_args()
    main(args)
