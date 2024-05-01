# Code for gathering seed functions

The pipeline for gathering seed functions is composed of three scripts, to be run in the following order:

1. `./generate_from_the_stack.py`: Gathers unfiltered seed functions with docstrings from The Stack v1.
2. `./high_quality_subset.py`: Transforms the seed functions using `autoimport`, and filters them by checking for a return statement and type-checking them with Pyright.
3. `./filter_dataset.py`: Further filters the seed functions by decontaminating the dataset, using StarCoder2-15B as a judge to remove bad examples, and using a set of static heuristics.

## 1. Generate from the Stack

In this step, we simply extract all functions from The Stack v1 (dedup) that have docstrings using tree-sitter. This is done by running the following command:

```bash
python3 generate_from_the_stack.py --push "<your hf name>/<your hf repo>"
```

This step may take a couple hours depending on your hardware. The resulting dataset will be pushed to the specified Hugging Face repository.

After running the command, we also run near-deduplication with MinHash, LSH, and Jaccard Similarity of 0.5.
We utilize the following repository for this step: https://github.com/ChenghaoMou/text-dedup

The dataset resulting from this step can be found here: https://huggingface.co/datasets/bigcode/stack-dedup-python-fns

## 2. High-quality subset

Here, we take the previously generated dataset and filter it and transform it using a set of heuristics.
We run the following steps:

1. We filter all functions which do not have a "return statement". This is done such that our execution
   filtering step in the instruction-generation pipeline does not have to deal with functions that do not return anything, which
   are hard to test. Another benefit is that this strengthens the type-checking step, as we can now validate the return type for all functions.
2. We infer imports for the functions using `autoimport`. Such that our standalone functions now correctly import any required modules.
3. We type-check each function using Pyright. This is done to filter any functions that may reference undefined identifiers or have static type errors (as detected by
   Pyright).

The dataset resulting from this step can be found here: https://huggingface.co/datasets/bigcode/python-stack-v1-functions-filtered

## 3. Filter dataset

Now, we further filter the dataset generated in in the previous with different methods:

1. We remove functions that contain a set of words that are likely to be bad examples (e.g. "TODO").
2. We also remove functions that import problematic packages, which can lead to issues in execution filtering (e.g. `os` or `sys`).
3. We remove functions which contains either solutions or prompts of benchmarks on which we evaluated the models.
4. We filter out functions that do not have any arguments, as these are likely to be bad examples in this constrained setting.
5. Finally, we utilize the base model StarCoder2-15B as a classifier to remove any examples that has bad documentation or low-quality code.

The dataset resulting from this step can be found here: https://huggingface.co/datasets/bigcode/python-stack-v1-functions-filtered-sc2
