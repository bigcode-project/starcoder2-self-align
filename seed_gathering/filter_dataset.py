import datasets
import os
from tree_sitter_parser import global_parser, LANGUAGE, does_have_return, make_parser
import benchmark_data
from tqdm import tqdm
import torch
import argparse
from vllm import LLM, SamplingParams
import random

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model', type=str,
                    default="bigcode/starcoder2-15b")
parser.add_argument('--batch-size', type=int, default=512)
parser.add_argument('--sample-size', type=int, default=None)
parser.add_argument('--num-gpus', type=int, default=1)
parser.add_argument('--content_col', type=str, default="content")
parser.add_argument('--push', type=str, required=True)
args = parser.parse_args()
random.seed(42)

FN_BLOCK_QUERY = LANGUAGE.query("""
(function_definition
  body: (block) @fn-block)
""")


def template_few_shot(code, answer, rationale):
    doc, code = py_extract_docstring(code)
    assert answer == "No" or answer == "Yes"
    prompt = f"""<issue_start>username_0: I have a function in Python and I'd like someone to check my description of this function.
I'm doing this so that I can write a good docstring for this function.

Here is the code for the function:
```py
{code}
```

Here is my description of this program:
```
{doc}
```

Do not attempt to execute the function or to judge its correctness.
Answer with "Yes" or "No" depending on if my description has enough information alone to re-implement the function.
Also, answer with "No" if the description does not match the function.<issue_comment>username_1: Sure, no problem. I will be able to help.
My answer is: {answer}

{rationale}

Upvotes: 200"""
    return prompt


FEW_SHOTS = [
    (
        '''def simple_scan_network():
    """
    Do a simple network scan, which only works if your network configuration
    is 192.168.1.x
    """
    base_ip = "192.168.1."
    addresses = ['127.0.0.1']

    for index in range(1, 255):
        addresses.extend([base_ip + str(index)])

    return addresses''',
        "No",
        "The simple_scan_network function you have provided seems to generate addresses that then would be used for a network scan, but does not actually perform it, unlike the function claims.",
    ),
    (
        '''import pandas


def coerce_integer(df):
    """
    Loop through the columns of a df, if it is numeric,
    convert it to integer and fill nans with zeros.
    This is somewhat heavy-handed in an attempt to force
    Esri to recognize sparse columns as integers.
    """
    # Numeric columns to not coerce to integer
    EXCEPT = ["latitude", "longitude", "zipCode"]

    def numeric_column_to_int(series):
        return (
            series.fillna(0).astype(int)
            if pandas.api.types.is_numeric_dtype(series) and series.name not in EXCEPT
            else series
        )

    return df.transform(numeric_column_to_int, axis=0)''',
        "Yes",
        "The docstring does seem to match the implementation! The function loops through the columns of a df and coerces it as explained.",
    ),
    ('''def __trans_df_into_dict(data):
    """Converte DataFrame to dictionary.

    Args:
        data (pandas.DataFrame): DataFrame.

    Returns:
        dict: Name dictionary.
    """
    data["en_name"] = data["en_name"].str.upper()
    data["en_name_f"] = data["en_name"].str.split(" ", expand=True)[0]
    data["en_name_l"] = data["en_name"].str.split(" ", expand=True)[1]
    data["jp_name_f"] = data["jp_name"].str.split("・", expand=True)[0]
    data["jp_name_l"] = data["jp_name"].str.split("・", expand=True)[1]
    fullname_dict = dict(zip(data["en_name"], data["jp_name"]))
    fname_dict = dict(zip(data["en_name_f"], data["jp_name_f"]))
    lname_dict = dict(zip(data["en_name_l"], data["jp_name_l"]))
    return fullname_dict, fname_dict, lname_dict''',
     "No",
     "The function__trans_df_into_dict  does indeed convert a dataframe into a dictionary, however, it converts various columns that were not described in the docstring.\nFor instance, nowhere in the docstring it mentions handling japanese characters or the name of the column.",
     ),
    (
        '''def inchesToMeters(inches):
    """Convert inches to meters."""
    return inches * 0.0254''',
        "Yes",
        "inchesToMeters is a very simple function, the doccstring explains concisely its purpose, which is of converting inches to meters.",
    ),
    ('''def square_crop(im, target_size=None):
  """ Crop image to `target_size`. If that's None the image is squared
  to the smallest size
  """

  w = im.size[0]
  h = im.size[1]

  target_size = target_size if target_size else min(w, h)

  dx = (w - target_size) / 2
  dy = (h - target_size) / 2

  return im.crop((dx, dy, dx + target_size, dy + target_size))''',
     "Yes",
     "Following the standard description for docstrings for functions and methods, the square_crop function description tells exactly what the function does."
     ),
    ('''def _setup_motifs_files(args):
    """convenience fn, make sure setup is same across
    multiplicity/orientation/spacing workflows
    """
    motifs_files = {}
    motifs_files["early"] = "{}/{}/ggr.scanmotifs.h5".format(
        args.inputs["inference"][args.cluster]["scanmotifs_dir"],
        args.inputs["inference"][args.cluster]["scanmotifs_early_dir"])
    motifs_files["mid"] = "{}/{}/ggr.scanmotifs.h5".format(
        args.inputs["inference"][args.cluster]["scanmotifs_dir"],
        args.inputs["inference"][args.cluster]["scanmotifs_mid_dir"])
    motifs_files["late"] = "{}/{}/ggr.scanmotifs.h5".format(
        args.inputs["inference"][args.cluster]["scanmotifs_dir"],
        args.inputs["inference"][args.cluster]["scanmotifs_late_dir"])

    return motifs_files''',
     "No",
     "The docstring for _setup_motifs_files just says this is a convenience function. There is definitely not enough information to re-implement this function from the docstring alone.",
     ),
    ('''def trip(u, v):
    """
    Returns the scalar triple product of vectors u and v and z axis.
    The convention is z dot (u cross v). Dotting with the z axis simplifies
    it to the z component of the u cross v
    The product is:
        positive if v is to the left of u, that is,
          the shortest right hand rotation from u to v is ccw
        negative if v is to the right of u, that is,
          the shortest right hand rotation from u to v is cw
        zero if v is colinear with u
    Essentially trip is the z component of the cross product of u x v
    """
    return (u[0] * v[1] - u[1] * v[0])''',
     "Yes",
     "The docstring for the trip function is very detailed and describes the function's purpose and the mathematical formula used to calculate the scalar triple product.",
     )
]


def prompt_fmt(code):
    doc, code = py_extract_docstring(code)
    random.shuffle(FEW_SHOTS)
    buf = ""
    for few in FEW_SHOTS:
        buf += template_few_shot(*few)
    buf += f"""<issue_start>username_0: I have a function in Python and I'd like someone to check my description of this function.
I'm doing this so that I can write a good docstring for this function.

Here is the code for the function:
```py
{code}
```

Here is my description of this program:
```
{doc}
```

Do not attempt to execute the function or to judge its correctness.
Answer with "Yes" or "No" depending on if my description has enough information alone to re-implement the function.
Also, answer with "No" if the description does not match the function.
Upvotes: 100<issue_comment>username_1: Sure, no problem. I will be able to help.
My answer is:"""
    return buf


def auto_dtype():
    if torch.cuda.is_bf16_supported():
        return "bfloat16"
    return "auto"


def chunkify(lst, n):
    chunks = []
    for i in range(0, len(lst), n):
        chunk = []
        for j in range(n):
            if i + j < len(lst):
                chunk.append(lst[i + j])
        chunks.append(chunk)
    return chunks


dataset = datasets.load_dataset(args.dataset, split="train")
print(f"Loaded {len(dataset)} examples. Running pre-filtering...")

BAD_WORDS = ["todo", "fixme", "bug"]
BAD_IMPORTS = ["argparse", "os", "subprocess", "sys", "setuptools",
               "distutils", "matplotlib", "seaborn"]
BAD_IMPORTS = [f"import {b}" for b in BAD_IMPORTS] + \
    [f"from {b}" for b in BAD_IMPORTS]
BAD_SUBSTRINGS = BAD_WORDS + BAD_IMPORTS

bench_filter = benchmark_data.filter_out()
all_bench = bench_filter["human_eval_docstrings"] + \
    bench_filter["human_eval_solutions"] + \
    bench_filter["mbpp_docstrings"] + \
    bench_filter["mbpp_solutions"]


def pre_filtering(ex):
    code = ex[args.content_col]
    code_bytes = code.encode('utf-8')

    # filter out bad substrings
    lower = code.lower()
    for word in BAD_SUBSTRINGS:
        if word in lower:
            return False

    for b in all_bench:
        if b in code:  # contaminated sample!
            return False

    # too many lines of code -- say 150
    lines = code.split("\n")
    if len(lines) > 150:
        return False

    # filter functions which don't have an argument
    # 1. find first def statement in lines
    # 2. check if contains ():
    for line in lines:
        if line.startswith("def ") and "():" in line:
            return False

    # filter out functions with no return statement
    parser = make_parser()
    if not does_have_return(code, parser=parser):
        return False

    try:
        tree = global_parser.parse(code_bytes)
        block, _ = FN_BLOCK_QUERY.captures(tree.root_node)[0]

        # get the docstring, filter if not a docstring
        exp = block.children[0]
        if not exp.type == 'expression_statement' and not exp.children[0].type == 'string':
            return False

        docstring = exp.children[0]
        docstring_text = docstring.text.decode('utf-8')
        if not docstring_text.startswith('"""') and not docstring_text.endswith('"""'):
            return False
    except Exception as e:
        print(f"Error in filtering: {e}")
        return False

    return True  # all good!


threads = os.cpu_count() - 1  # type: ignore
dataset = dataset.filter(pre_filtering, num_proc=threads)

model = LLM(args.model, dtype=auto_dtype(),
            gpu_memory_utilization=0.95, tensor_parallel_size=args.num_gpus)
tokenizer = model.get_tokenizer()

if args.sample_size is not None:
    dataset = dataset.shuffle()
    dataset = dataset.select(range(args.sample_size))


print(f"Now running stage 3 filtering on {len(dataset)} examples...")


def unindent(s):
    lines = s.splitlines()
    non_blank_lines = [line for line in lines if line.strip()]
    min_indent = min(len(line) - len(line.lstrip())
                     for line in non_blank_lines) if non_blank_lines else 0
    unindented_lines = [line[min_indent:] if len(
        line) >= min_indent else line for line in lines]
    return '\n'.join(unindented_lines)


def py_extract_docstring(code):
    first_doc = code.find('"""')
    assert first_doc != -1
    first_doc = first_doc + 3
    second_doc = code[first_doc+1:].find('"""')
    assert second_doc != -1
    second_doc = second_doc + first_doc + 1
    doc = code[first_doc:second_doc]
    doc = unindent(doc).strip()
    code = code[:first_doc-3] + code[second_doc+3:]
    return doc, code


# this is such a hack, but it works
dummy = 'def dummy(): \n    """\n    """\n pass'
dummy_prompt = prompt_fmt(dummy)
few_shot_toks = len(tokenizer.encode(
    dummy_prompt)) - len(tokenizer.encode(dummy))
print(f"Few-shot prompt has {few_shot_toks} tokens")
prompts = []
for ex in tqdm(dataset, total=len(dataset), desc="Generating prompts"):
    code = ex[args.content_col]
    toks = len(tokenizer.encode(code)) + few_shot_toks
    if toks > 16380:
        print(f"Skipping example with {toks} tokens")
        # to skip, just add dummy prompt
        prompts.append(dummy_prompt)
        continue
    p = prompt_fmt(code)
    prompts.append(p)

responses = []
for chunk in tqdm(chunkify(prompts, args.batch_size), desc="Generating responses"):
    outs = model.generate(chunk, SamplingParams(
        temperature=0.0, stop="\n", max_tokens=5))
    contents = [o.outputs[0].text for o in outs]
    for c in contents:
        yes_count = c.lower().count("yes")
        no_count = c.lower().count("no")
        if yes_count > no_count:
            responses.append(True)
        elif yes_count < no_count:
            responses.append(False)
        else:
            # default to No
            responses.append(False)


new_ds = dataset.filter(  # horrible hack!
    lambda ex, i: responses[i] and "def dummy()" not in ex[args.content_col], with_indices=True)
print(f"Filtered {len(dataset) - len(new_ds)} examples")
new_ds.push_to_hub(args.push, private=True)
