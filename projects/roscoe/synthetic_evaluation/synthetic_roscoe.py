#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate perturbed synthetic dataset.

Usage:
python projects/roscoe/synthetic_evaluation/synthetic_roscoe.py
"""
import json
import os
from typing import List

from projects.roscoe.score import (
    EMBEDDING_MODEL_NAMES,
    EMBEDDING_MODELS,
    SEQ_EMB_MODEL_TYPES,
    Chain,
    Evaluator,
    REASONING_SCORES,
    SCORE_GROUPS,
    select_semantic_scores,
    select_inference_scores,
    select_language_scores,
    SENT_TRANS,
)
from projects.roscoe.utils import (
    print_and_reset_max_gpu_memory,
    save_scores,
)

from parlai.core.params import ParlaiParser

DEFAULT_FILE_PATH = f"./projects/roscoe/roscoe_data/synthetic_50%/aqua_synthetic/"
DEFAULT_OUTPUT_PATH = f"./projects/roscoe/scores/"


class SyntheticChain(Chain):
    def __init__(self, line: List[str]) -> None:
        self.chain = self.parse_chain(line)

    def parse_chain(self, chain: List[str]) -> List[str]:
        """
        Change formatting Returns list of steps in reasoning chain.
        """
        steps = []
        for step in chain:
            if len(step) > 0:
                step = step.replace("\t", " ")
                # "dots" are not always generated, to ensure uniformity
                while len(step) > 0 and (step[-1] == '.' or step[-1] == ' '):
                    step = step[:-1]
                if len(step) == 0:
                    print(f"WARNING: Found step without text in chain {chain}")
                else:
                    steps.append(step)
            else:
                print(f"WARNING: Found empty step in chain {chain}")
        return steps


class SyntheticEvaluator(Evaluator):
    def __init__(
        self,
        model_type: str,
        transformer_model: str,
        discourse_batch: int,
        coherence_batch: int,
        **kwargs,
    ) -> None:
        super().__init__(
            hypos=[],
            context=[],
            references=[],
            model_type=model_type,
            transformer_model=transformer_model,
            discourse_batch=discourse_batch,
            coherence_batch=coherence_batch,
            **kwargs,
        )

    def update_evaluator(self, in_file: str):
        hypothesises = []
        refs = []
        contexts = []
        with open(in_file) as _f:
            for line in _f:
                jline = json.loads(line)
                h_chain = SyntheticChain(line=jline["dialog"][0][0]["steps"])
                r_chain = SyntheticChain(line=jline["dialog"][0][0]["original_steps"])
                context = SyntheticChain(
                    line=jline["dialog"][0][0]["question"].split(". ")
                )
                hypothesises.append(h_chain)
                refs.append(r_chain)
                contexts.append(context)
        super().set_hypos(hypothesises)
        super().set_context(contexts)
        super().set_references(refs)


def score_and_save(
    evaluator: SyntheticEvaluator,
    subdirectory: str,
    root: str,
    data_filename: str,
    score_types: List[str],
    out_dir: str,
) -> None:
    d = root.split('/')[-1]
    if len(d) == 0:
        d = root.split('/')[-2]

    out_path = os.path.join(
        out_dir,
        subdirectory,
        f"scores_{d + '_' + data_filename.split('.')[0]}.tsv",
    )
    if os.path.exists(out_path):
        print(f"Score file for {data_filename} already exists. Skipping.")
        return

    print(f"Evaluating {os.path.join(root, data_filename)}")
    evaluator.update_evaluator(os.path.join(root, data_filename))
    scores = evaluator.evaluate(score_types=score_types)
    save_scores(scores, out_path)
    print_and_reset_max_gpu_memory()


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument(
        '--dataset-path',
        '-p',
        type=str,
        required=False,
        default=DEFAULT_FILE_PATH,
        help='Path to files with predictions',
    )
    parser.add_argument(
        '--suffix',
        '-s',
        type=str,
        required=False,
        default="test",
        help='File suffix to match',
    )
    parser.add_argument(
        '--scores',
        type=str,
        nargs="*",
        default=REASONING_SCORES,
        choices=REASONING_SCORES,
        help='Path to files with predictions',
    )
    parser.add_argument(
        '--model-type',
        '-t',
        type=str,
        required=False,
        default=SENT_TRANS,
        choices=SEQ_EMB_MODEL_TYPES,
        help='Model type for embedding sequences.',
    )
    parser.add_argument(
        '--model-name',
        '-m',
        type=str,
        required=False,
        default="all-mpnet-base-v2",
        choices=EMBEDDING_MODELS,
        help='Transformer model name for embeddings. Must be compatible with model_type',
    )
    parser.add_argument(
        '--ppl-model-name',
        type=str,
        required=False,
        default="gpt2-large",
        help='Transformer HuggingFace model name for calculating perplexity-based metrics.',
    )
    parser.add_argument(
        '--discourse-batch',
        '-db',
        type=int,
        required=False,
        default=64,
        help='Batch size for discourse calculation',
    )
    parser.add_argument(
        '--coherence-batch',
        '-cb',
        type=int,
        required=False,
        default=8,
        help='Batch size for coherence calculation',
    )
    parser.add_argument(
        '--ppl-batch',
        type=int,
        required=False,
        default=16,
        help='Batch size for perplexity calculation',
    )
    parser.add_argument(
        '--grammar-batch',
        type=int,
        required=False,
        default=8,
        help='Batch size for grammar score calculation',
    )
    parser.add_argument(
        '--file-name-filter',
        type=str,
        required=False,
        default=None,
        help='If specified, only scores files whose name contains this string.',
    )
    parser.add_argument(
        '--output-directory',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Where to save the scores.',
    )

    opt = parser.parse_args()
    evaluator = SyntheticEvaluator(
        score_types=opt["scores"],
        model_type=opt["model_type"],
        transformer_model=opt["model_name"],
        ppl_model=opt["ppl_model_name"],
        discourse_batch=opt["discourse_batch"],
        coherence_batch=opt["coherence_batch"],
        ppl_batch=opt["ppl_batch"],
        grammar_batch=opt["grammar_batch"],
    )
    fname_filter = opt['file_name_filter']
    model_nickname = opt["model_name"].split('/')[-1]
    for root, _dirnames, filenames in os.walk(opt['dataset_path']):
        for filename in filenames:
            if opt["suffix"] not in filename or ".swp" in filename:
                continue
            if fname_filter and fname_filter not in filename:
                print(f"Skipping due to file name filter: {filename}")
                continue
            score_groups = {
                model_nickname: select_semantic_scores(opt["scores"]),
                'inference': select_inference_scores(opt["scores"]),
                'language': select_language_scores(opt["scores"]),
            }
            assert all(
                k in EMBEDDING_MODEL_NAMES + SCORE_GROUPS for k in score_groups.keys()
            ), "All subdirectories should map to embedding model names or score groups."
            for subdirectory, score_subset in score_groups.items():
                if len(score_subset) == 0:
                    continue
                score_and_save(
                    evaluator=evaluator,
                    subdirectory=subdirectory,
                    root=root,
                    data_filename=filename,
                    score_types=score_subset,
                    out_dir=opt['output_directory'],
                )
