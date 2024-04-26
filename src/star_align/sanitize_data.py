import os
import random
import sys

from star_align.utils import (
    find_code_blocks,
    read_jsonl,
    remove_comments_from_code_blocks,
    write_jsonl,
)

src = sys.argv[1]
tgt = sys.argv[2]

xs = read_jsonl(src)
print("Before:", len(xs))

random.seed(0)

removing_tests = os.getenv("NO_TESTS") is not None
removing_explanation = os.getenv("NO_EXPL") is not None
removing_reasoning = os.getenv("NO_REASONING") is not None
removing_comments = os.getenv("NO_COMMENTS") is not None
removing_incomplete = os.getenv("NO_INCOMPLETE") is not None
codeonly = os.getenv("CODEONLY") is not None
augmentation_prob = float(os.getenv("AUGMENTATION", 0.0))
keep_raw_format = os.getenv("RAW") is not None
smart = os.getenv("SMART") is not None

incomplete_substrings = [
    "todo",
    "fixme",
    "write your code here",
    "your code here",
    "your code goes here",
    "notimplemented",
]

if removing_tests:
    print("Removing tests")
if removing_explanation:
    print("Removing explanation")
if removing_reasoning:
    print("Removing reasoning")
if removing_comments:
    print("Removing comments")
if removing_incomplete:
    print("Removing incomplete")
if codeonly:
    print("Code only")
if augmentation_prob > 0:
    print("Augmentation prob:", augmentation_prob)
if keep_raw_format:
    print("Keeping raw format")
if smart:
    removing_comments = True
    augmentation_prob = 0.5
    print("Smart mode")


def filter_x(x):
    response = x["response"]
    tags = ["[Tests]", "[Reasoning]", "[Explanation]", "[Implementation]"]
    # response = "".join(l for l in response.splitlines(keepends=True) if l.strip().startswith(""))
    if any(response.count(tag) != 1 for tag in tags):
        return False
    tests_index = response.index("[Tests]")
    explanation_index = response.index("[Explanation]")
    reasoning_index = response.index("[Reasoning]")
    implementation_index = response.index("[Implementation]")
    if not (reasoning_index < implementation_index < explanation_index < tests_index):
        return False
    reasoning = response[
        reasoning_index + len("[Reasoning]") : implementation_index
    ].strip()
    explanation = response[
        explanation_index + len("[Explanation]") : tests_index
    ].strip()
    implementation = response[
        implementation_index + len("[Implementation]") : explanation_index
    ].strip()
    codeblocks = find_code_blocks(implementation, "python")
    if len(codeblocks) == 0:
        return False
    if codeonly:
        code = "\n\n".join(codeblocks)
        implementation = f"```python\n{code}\n```"
    tests = response[tests_index + len("[Tests]") :].strip()
    # tests = tests.split("\n[")[0].split("\n##")[0].strip()
    tests_blocks = find_code_blocks(tests, "python")
    if len(tests_blocks) != 1 or tests.count("```") != 2:
        if os.getenv("DEBUG"):
            breakpoint()
        return False
    assert tests.count("```") == 2
    index = tests.rindex("```") + 3
    tests_prefix = tests[:index]
    # tests_suffix = tests[index:]
    # tests_suffix = tests_suffix.split("\n[")[0].split("\n#")[0].rstrip()
    # tests = tests_prefix + tests_suffix
    # remove NL after test block
    tests = tests_prefix
    strange_index = next(
        (idx for idx, l in enumerate(tests.splitlines()) if l.startswith("/")),
        None,
    )
    if strange_index is not None:
        # print("MD Index:", strange_index)
        tests = "".join(tests.splitlines(keepends=True)[:strange_index]).strip()
    if "assert" not in tests or all(
        l.startswith("def")
        or l.startswith("class")
        or l.startswith("import")
        or l.startswith("from")
        for l in tests_blocks[0].splitlines()
        if len(l) > 0 and l[0].isalpha()
    ):
        return False

    global removing_tests, removing_explanation, removing_reasoning, removing_incomplete

    if smart:
        removing_tests = "test" not in x["instruction"].lower()
        removing_explanation = random.random() < 0.5
        removing_reasoning = random.random() < 0.5
        removing_incomplete = True

    if keep_raw_format:
        contents = [
            "[Reasoning]\n" + reasoning,
            "[Implementation]\n" + implementation,
            "[Explanation]\n" + explanation,
            "[Tests]\n" + tests,
        ]
    else:
        contents = [reasoning, implementation, explanation, tests]

    if removing_incomplete:
        if any(
            substring in x["response"].lower() for substring in incomplete_substrings
        ):
            return False

    if removing_tests:
        contents.remove(tests)
    if removing_explanation:
        contents.remove(explanation)
    if removing_reasoning:
        contents.remove(reasoning)
    x["response"] = "\n\n".join(contents)

    tests_block = tests_blocks[0]
    lines = tests_block.splitlines()
    if all(l.startswith("assert") for l in lines):
        ks = [1, 2, 3, 4, 5]
        assertions = random.sample(lines, k=min(random.choice(ks), len(lines)))
        assertion = "\n".join(assertions)
        assertion_term = "assertion" + ("s" if len(assertions) > 1 else "")
    else:
        assertion = tests_block
        assertion_term = "test case"
    if (
        augmentation_prob > 0
        and "assert" in assertion
        # 5 lines augmented block max
        and len(assertion.splitlines()) <= 5
        and random.random() < augmentation_prob
        and "assert" not in x["instruction"]
        and "for example" not in x["instruction"].lower()
        and (not smart or "test" not in x["instruction"].lower())
    ):
        # if smart:
        #     contents.remove(tests)
        # else:
        assert removing_tests
        assert "assert" in assertion
        assertion_str = (
            f"Your code should pass the following {assertion_term}:\n```python\n"
            + assertion.strip()
            + "\n```"
        )
        x["instruction"] = f"{x['instruction']}\n\n{assertion_str}"
    if removing_comments:
        x["response"] = remove_comments_from_code_blocks(x["response"])
    # for tag in tags:
    #     x["response"] = x["response"].replace(f"{tag}\n", "")
    return True


xs = [x for x in xs if filter_x(x)]
print("After:", len(xs))
write_jsonl(tgt, xs)
