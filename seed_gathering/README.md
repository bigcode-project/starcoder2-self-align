# Code for gathering seed functions

- `./generate_from_the_stack.py`: Gathers unfiltered seed functions with docstrings from The Stack v1.
- `./high_quality_subset.py`: Filters the seed functions using a set of heuristics, including type-checking, parsing, and import inference.
- `./filter_dataset.py`: Further filters the seed functions by decontaminating the dataset and using StarCoder2-15B as a judge.
