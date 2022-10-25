#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
import pathlib
from nltk.tokenize import sent_tokenize
from typing import Callable, Dict, Iterable, List, Tuple

import torch


def save_scores(score_dict: Dict, out_path: str) -> None:
    # create destination directory if not exists
    pathlib.Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    score_list = list(score_dict.keys())
    with open(out_path, 'w') as output_file:
        n_scores = len(score_list)
        out_line = "{:<8} " + " ".join(["{:<15}" for i in range(n_scores)])
        print(out_line.format('ID', *score_list), file=output_file)
        n_samples = len(score_dict[score_list[0]])
        for i in range(n_samples):
            scores = []
            for score in score_list:
                scores.append(score_dict[score][i])
            print(out_line.format(i, *scores), file=output_file)
    print(f"Scores written to {out_path}")


def print_and_reset_max_gpu_memory() -> None:
    max_gpu_mem_alloc = int(torch.cuda.max_memory_allocated() // 1e6)
    print(f"Max GPU Memory Allocated: {max_gpu_mem_alloc} MB")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def cosine_similarity_scaled(list1: np.ndarray, list2: np.ndarray) -> float:
    """
    Normalized cosine similarity for *normalized* embeddings.

    Normalized cosine similarity takes values from [0;1]
    """
    cosine_sim = np.dot(list1, list2) / (np.linalg.norm(list1) * np.linalg.norm(list2))
    return (1.0 + cosine_sim) / 2.0


def embedding_alignment(ref_emb: np.ndarray, hypo_emb: np.ndarray) -> List[float]:
    """
    Return embedding matching alignment for each item in hypo_emb
    ref_emb: list of reference embeddings
    hypo_emb: list oh hypothesises embeddings
    """
    scores = []
    for he in hypo_emb:
        # some embeddings can be empty. For example, for latex-style equations, or empty string
        if len(he) > 0:
            out = [cosine_similarity_scaled(he, re) for re in ref_emb if len(re) > 0]
            if len(out) > 0:
                scores.append(max(out))
    return scores


def al_mean(alignment_scores) -> float:
    if len(alignment_scores) == 0:
        return 0.0
    return sum(alignment_scores) / len(alignment_scores)


def split_gsm8k_gpt3_generations_to_steps(reasoning: str) -> List[str]:
    """
    This logic is copied directly from the code that parsed GSM8K generations into steps
    for annotation.
    """
    return [
        split
        for s in sent_tokenize(reasoning)
        for split in s.split("\n")
        if len(split) > 0
    ]


def assert_all_elements_same_length(
    elements: Iterable,
    error_msg: str = 'not all elements have the same length',
) -> int:
    """
    Asserts that all elements in the iterable have the same length.

    Can be useful when you have a list of lists representing rows or columns, for
    example. Returns the length.
    """
    unique_lengths = set(len(l) for l in elements)
    assert len(unique_lengths) == 1, f"{error_msg} | {unique_lengths}"
    return list(unique_lengths)[0]


def split_list(
    input_list: Iterable[str],
    include_condition: Callable[[str], bool],
) -> Tuple[List[str], List[str]]:
    """
    Splits a list into two based on a condition applied to each element.
    """
    matching_list = [x for x in input_list if include_condition(x)]
    other_list = [x for x in input_list if x not in matching_list]
    return matching_list, other_list


def ordered_union(list_of_lists: List[List[str]]) -> List[str]:
    """
    Unpacks a list of lists, ensuring there are no duplicates in the final list.
    """
    union_list = []
    for l in list_of_lists:
        for item in l:
            if item not in union_list:
                union_list.append(item)
    return union_list
