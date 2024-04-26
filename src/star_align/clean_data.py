from star_align.utils import read_jsonl, write_jsonl
import sys
import re

dataset = read_jsonl(sys.argv[1])

def contains_chinese(s):
    return bool(re.search(r'[\u4e00-\u9fff]', s))

chosen = []
rejected = []
for example in dataset:
    if "code snippet" in example["instruction"] or contains_chinese(example["instruction"] + example["response"]):
        rejected.append(example)
    else:
        chosen.append(example)

print(f"Removed {len(dataset) - len(chosen)} examples")
write_jsonl(sys.argv[2], chosen)
