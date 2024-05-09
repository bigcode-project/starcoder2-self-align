import json
import argparse
import random
import multiprocessing as mp
import os
import re
from collections import defaultdict
from typing import Any, Callable, List

import click
import datasets
import numpy as np
from tqdm import tqdm

import pickle  # nosec
from collections import Counter
from pathlib import Path
from itertools import tee

from scipy.integrate import quad as integrate

import hashlib
import struct
from hashlib import md5
from hashlib import sha256

import xxhash
from xxhash import xxh3_64
from xxhash import xxh3_64_digest
from xxhash import xxh3_128
from xxhash import xxh3_128_digest


parser = argparse.ArgumentParser()
# IO Args
parser.add_argument("--data_files", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
parser.add_argument("--num_proc", type=int, default=os.cpu_count())
# Meta Args
parser.add_argument("--column", type=str, required=True)
parser.add_argument("--batch_size", type=int, default=10_000)
# MinHash Args
parser.add_argument("--ngram", type=int, default=5)
parser.add_argument("--min_length", type=int, default=5)
parser.add_argument("--ignore_empty", type=bool, default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_perm", type=int, default=250)
parser.add_argument("--threshold", type=float, default=0.7)
parser.add_argument("--b", type=int, default=None)
parser.add_argument("--r", type=int, default=None)
parser.add_argument("--hash_func", type=str, default="sha1")
parser.add_argument("--hash_bits", type=int, default=64)
args = parser.parse_args()


def ngrams(sequence: List[str], n: int, min_length: int = 5):
    """
    Return the ngrams generated from a sequence of items, as an iterator.

    This is a modified version of nltk.util.ngrams.
    """
    if len(sequence) < min_length:
        return []
    if len(sequence) < n:
        return [tuple(sequence)]
    iterables = tee(iter(sequence), n)
    for i, sub_iterable in enumerate(iterables):
        for _ in range(i):
            next(sub_iterable, None)
    return zip(*iterables)


class UnionFind:
    """
    A data structure for maintaining disjoint sets. This helps build connected components for given duplicate pairs.
    """

    def __init__(self):
        self.parent = {}
        # Counter is a subclass of dict with slightly different python and c implementations
        # you can think of it as an optimized defaultdict(int)
        self.rank = Counter()

    def find(self, x):
        try:
            # path compression
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
        except KeyError:
            # KeyError happens if x not in parent
            self.parent[x] = x
        finally:
            return self.parent[x]

    def union(self, x, y):
        px = self.find(x)
        py = self.find(y)

        # If both elements are already in the same set, do nothing
        # The line in original UnionFind `self.parent[px] = self.parent[py] = min(px, py)` is redundant when px == py
        if px == py:
            return

        if self.rank[px] == self.rank[py]:
            # If ranks are equal, choose one as the new root and increment its rank
            # with few duplicates this is likely to be the most common case
            self.parent[py] = px
            self.rank[px] += 1
        # otherwise, assume that leftside is more likely to be higher rank
        # Attach the smaller rank tree under the root of the larger rank tree
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[px] = py

    def reset(self):
        self.parent = {}
        self.rank = Counter()

    def dump(self, path: str | Path, id2id=None):
        if id2id is not None:
            new_uf = UnionFind()
            for i in self.parent:
                new_uf.union(id2id[i], id2id[self.find(i)])
        else:
            new_uf = self

        with open(path, "wb") as f:
            pickle.dump(new_uf, f, protocol=pickle.HIGHEST_PROTOCOL)


RNG = np.random.RandomState(args.seed)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()

SIGNATURE_COLUMN = "__signatures__"
INDEX_COLUMN = "__index__"
CLUSTER_COLUMN = "__cluster__"

# for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
# is not copied to child processes as long as it is not modified.
mp.set_start_method("fork", force=True)
uf = UnionFind()


def sha1_hash(data: bytes, d: int = 32) -> int:
    """
    Generate a d-bit hash value from the given data.
    """
    if d == 32:
        return struct.unpack(
            "<I", hashlib.sha1(data, usedforsecurity=False).digest()[:4]
        )[0]
    if d == 64:
        return struct.unpack(
            "<Q", hashlib.sha1(data, usedforsecurity=False).digest()[:8]
        )[0]
    # struct is faster but does not support arbitrary bit lengths
    return int.from_bytes(
        hashlib.sha1(data, usedforsecurity=False).digest()[: d // 8], byteorder="little"
    )


def xxh3_16hash(data: bytes, seed: int = 0) -> int:
    """
    Generate a 16-bit xxhash based hash value from the given data.
    As of python xxhash 3.3.0 (and since 0.3.0) outputs in big-endian.
    This is useful as a special purpose xxhash when you only want 16 bits.
    bit masked xxh3_64 hashes are faster than xxh32 in modern systems.
    """
    return xxhash.xxh3_64_intdigest(data, seed) & 0xFFFF


def xxh3_32hash(data: bytes, seed: int = 0) -> int:
    """
    Generate a 32-bit xxhash based hash value from the given data.
    As of python xxhash 3.3.0 (and since 0.3.0) outputs in big-endian.
    This is useful as a special purpose xxhash when you only want 32bits.
    bit masked xxh3_64 hashes are faster than xxh32 in modern systems.
    """
    return xxhash.xxh3_64_intdigest(data, seed) & 0xFFFFFFFF


def optimal_param(
    threshold: float,
    num_perm: int,
    false_positive_weight: float = 0.5,
    false_negative_weight: float = 0.5,
):
    """
    Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
    of probabilities of false positive and false negative, taken from datasketch.
    """

    def false_positive_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - s ** float(r)) ** float(b)

        a, _ = integrate(proba, 0.0, threshold)
        return a

    def false_negative_area(threshold: float, b: int, r: int):
        """Source: `datasketch.lsh`"""

        def proba(s):
            return 1 - (1 - (1 - s ** float(r)) ** float(b))

        a, _ = integrate(proba, threshold, 1.0)
        return a

    min_error = float("inf")
    opt = (0, 0)
    for b in range(1, num_perm + 1):
        max_r = int(num_perm / b)
        for r in range(1, max_r + 1):
            fp = false_positive_area(threshold, b, r)
            fn = false_negative_area(threshold, b, r)
            error = fp * false_positive_weight + fn * false_negative_weight
            if error < min_error:
                min_error = error
                opt = (b, r)
    return opt


def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    min_length: int,
    hashranges: list[tuple[int, int]],
    permutations: np.ndarray,
    hash_func: Callable,
    dtype: type,
    max_hash: np.uint,
    modulo_prime: np.uint,
) -> dict[str, Any]:
    """
    Calculate hash values for the content.
    """
    # a, b are each np.ndarray arrays containing {num_perm} pairs of random numbers used for building new hashes
    # the formula is a * x(base hash of each shingle) + b
    a, b = permutations
    # split content on whitespace (NON_ALPHA regex), tokenize with ngrams(), and join these n-grams into a single space separated string.
    # we then convert to lower case and then bytestrings which is then hashed. Only unique hashed n-grams are left.
    tokens: set[bytes] = {
        bytes(" ".join(t).lower(), "utf-8")
        for t in ngrams(NON_ALPHA.split(content.lower()), ngram_size, min_length)
    }

    hashvalues: np.ndarray = np.array(
        [hash_func(token) for token in tokens], dtype=dtype
    ).reshape(len(tokens), 1)
    # Permute the hash values to produce new universal hashes
    # Element-wise multiplication with 'hashvalues' and a (non 0 random value) and then adding b
    # Then, take modulo 'MODULO_PRIME' and bitwise_and with 'MAX_HASH' to keep only the necessary bits.
    hashvalues = (hashvalues * a + b) % modulo_prime & max_hash
    # this part is where the name "min" of minhash comes from
    # this stacks all the hashes and then takes the minimum from each column
    masks: np.ndarray = np.full(shape=num_perm, dtype=dtype, fill_value=max_hash)
    hashvalues = np.vstack([hashvalues, masks]).min(axis=0)
    # Originally, byteswap was done for speed. Testing show it has a negligible impact
    # keeping  for backward compatibility, even though theoretically and empirically
    # it doesnt matter if it is there or not. github.com/ekzhu/datasketch/issues/114
    Hs: list[bytes] = [
        bytes(hashvalues[start:end].byteswap().data) for start, end in hashranges
    ]
    return {SIGNATURE_COLUMN: Hs, INDEX_COLUMN: idx}


def main():
    global uf
    uf.reset()
    HASH_BITS: int = args.hash_bits
    HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16 bit config does not use a mersenne prime.
        # The original reason for using mersenne prime was speed.
        # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(HASH_BITS, HASH_CONFIG[64])

    match args.hash_func:
        case "sha1":

            def hash_func(byte_data):
                return sha1_hash(byte_data, d=min(HASH_BITS, 32))

        case "xxh3":
            if HASH_BITS == 16:
                hash_func = xxh3_16hash
            else:
                hash_func = xxh3_32hash

    if args.b is not None and args.r is not None:
        B, R = args.b, args.r
    else:
        # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        # of probabilities of false positive and false negative, taken from datasketch.
        B, R = optimal_param(
            args.threshold,
            args.num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES = [defaultdict(set) for _ in range(B)]

    PERMUTATIONS = (
        RNG.randint(
            1, MODULO_PRIME, size=(args.num_perm,), dtype=DTYPE
        ),  # a is a multiplier so should not be 0
        RNG.randint(0, MODULO_PRIME, size=(args.num_perm,), dtype=DTYPE),  # b
    )

    # Loading
    data_files_list = [x.strip() for x in args.data_files.split(",")]
    ds = datasets.load_dataset("json", data_files=data_files_list, split="train")
    ds = ds.map(
        lambda x, i: {INDEX_COLUMN: i}, with_indices=True, num_proc=args.num_proc
    )

    if args.ignore_empty:
        ds_rest = ds.filter(lambda x: len(x[args.column].strip()) == 0)
        ds = ds.filter(lambda x: len(x[args.column].strip()) > 0)

    ds = ds.filter(
        lambda x: len(NON_ALPHA.split(x[args.column].lower())) >= args.min_length,
        num_proc=args.num_proc,
    )

    LEN_DATASET = len(ds)
    if args.ignore_empty:
        LEN_DATASET += len(ds_rest)

    # MinHashing
    embedded = ds.map(
        function=embed_func,
        fn_kwargs={
            "num_perm": args.num_perm,
            "hashranges": HASH_RANGES,
            "ngram_size": args.ngram,
            "min_length": args.min_length,
            "permutations": PERMUTATIONS,
            "hash_func": hash_func,
            "dtype": DTYPE,
            "max_hash": MAX_HASH,
            "modulo_prime": MODULO_PRIME,
        },
        input_columns=[args.column, INDEX_COLUMN],
        remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
        num_proc=args.num_proc,
        with_indices=False,
        desc="Fingerprinting...",
    )
    LEN_EMBEDDED = len(embedded)
    NUM_SHARDS = np.ceil(LEN_EMBEDDED / args.batch_size).astype(int)

    # Clustering
    edges = []
    for i in tqdm(
        range(0, NUM_SHARDS),
        dynamic_ncols=True,
        desc="Iterating MinHashes...",  # noqa: E501
    ):
        embedded_shard = embedded.shard(
            num_shards=NUM_SHARDS,
            index=i,
            contiguous=True,
            writer_batch_size=args.batch_size,
        )
        for key, Hs in zip(
            embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]
        ):
            for i, H in enumerate(Hs):
                HASH_TABLES[i][H].add(key)

    print(f"Number of clusters: {len(HASH_TABLES)}")
    for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
        # cluster: Set[int]
        for cluster in table.values():
            if len(cluster) <= 1:
                continue
            idx = min(cluster)
            for x in cluster:
                edges.append((x, idx))
                uf.union(x, idx)
    print(f"Number of edges: {len(set(edges))}")

    # Filtering
    ds = ds.map(
        function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
        with_indices=False,
        num_proc=args.num_proc,
        new_fingerprint=str(random.getrandbits(128)),
        desc="Finding clusters...",
    )
    # This is where the deduplication happens
    # Since there is no easy groupby in datasets
    # I will use this simple filter for now
    final_data = ds.filter(
        function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
        with_indices=False,
        num_proc=args.num_proc,
        desc="Filtering clusters...",
    )
    if args.ignore_empty and len(ds_rest) > 0:
        final_data = datasets.concatenate_datasets([ds_rest, final_data])

    # Saving
    final_data = final_data.remove_columns([CLUSTER_COLUMN, INDEX_COLUMN])
    final_data.to_json(args.output)
    print("Before:", LEN_DATASET)
    print("After:", len(final_data))

    # Cleaning
    ds.cleanup_cache_files()
    final_data.cleanup_cache_files()


if __name__ == "__main__":
    main()
