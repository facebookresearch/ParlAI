#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate dataset of generated chains-of-resoning.

Usage:
python projects/roscoe/roscoe.py
"""
import json
import os
from typing import List

from nltk.tokenize import sent_tokenize

from projects.roscoe.score import (
    SEQ_EMB_MODEL_TYPES,
    Chain,
    Evaluator,
    REASONING_SCORES,
    UNSUPERVISED_SCORES,
    SENT_TRANS,
)
from projects.roscoe.utils import (
    print_and_reset_max_gpu_memory,
    save_scores,
    split_gsm8k_gpt3_generations_to_steps,
)

from parlai.core.params import ParlaiParser

DEFAULT_INPUT_PATH = f"./projects/roscoe/roscoe_data/generated/"
DEFAULT_OUTPUT_PATH = f"./projects/roscoe/scores/"

DATASETS = ["drop", "esnli", "cosmos", "gsm8k", "semeval"]


class ReasoningSteps(Chain):
    def __init__(self, line: str, type="regular") -> None:
        self.chain = self.parse_chain(line, type=type)

    def parse_chain(self, chain: str, type: str) -> List[str]:
        """
        Change formatting.

        Returns list of steps in reasoning chain.
        """
        if type == "gsm8k_ref":
            return chain.split("IGNORE THIS. Ground truth here for reference. ")[
                1
            ].split('\n')
        elif type == "gsm8k_hypo":
            return split_gsm8k_gpt3_generations_to_steps(reasoning=chain)
        elif type == "regular":
            return sent_tokenize(chain)
        else:
            raise NotImplementedError(f"{type} chain type is not supported")


class ReasoningEvaluator(Evaluator):
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
        contexts = []
        refs = []
        with open(in_file) as _f:
            for line in _f:
                jline = json.loads(line)
                h_chain = ReasoningSteps(line=jline["gpt-3"])
                context = ReasoningSteps(
                    line=jline["premise"] + " " + jline["hypothesis"]
                )
                if "gsm8k" in in_file:
                    context = ReasoningSteps(line=jline["premise"])
                    h_chain = ReasoningSteps(line=jline["gpt-3"], type="gsm8k_hypo")
                    r_chain = ReasoningSteps(line=jline["hypothesis"], type="gsm8k_ref")
                    refs.append(r_chain)
                hypothesises.append(h_chain)
                contexts.append(context)
                if "esnli" in in_file:
                    r_chain = ReasoningSteps(
                        line=jline["explanation_1"]
                        + " "
                        + jline["explanation_2"]
                        + " "
                        + jline["explanation_3"]
                    )
                    refs.append(r_chain)
        super().set_hypos(hypothesises)
        super().set_context(contexts)
        super().set_references(refs)


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument(
        '--dataset-path',
        '-p',
        type=str,
        required=False,
        default=DEFAULT_INPUT_PATH,
        help='Path to files with predictions',
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs="+",
        default=DATASETS,
        help='Which datasets to score.',
    )
    parser.add_argument(
        '--suffix',
        '-s',
        type=str,
        required=False,
        default="json",
        help='File suffix to match',
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
        default=16,
        help='Batch size for coherence calculation',
    )
    parser.add_argument(
        '--scores',
        type=str,
        nargs="*",
        default=REASONING_SCORES,
        choices=REASONING_SCORES,
        help=(
            'Scores to calculate. If the data is incompatible with a specified score '
            '(e.g. no reference is available) the score will be ignored.'
        ),
    )
    parser.add_argument(
        '--output-directory',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Where to save the scores.',
    )

    opt = parser.parse_args()
    evaluator = ReasoningEvaluator(
        score_types=opt['scores'],
        model_type=opt["model_type"],
        transformer_model=opt["model_name"],
        ppl_model=opt["ppl_model_name"],
        discourse_batch=opt["discourse_batch"],
        coherence_batch=opt["coherence_batch"],
    )
    for root, _dirnames, filenames in os.walk(opt['dataset_path']):
        for filename in filenames:
            if opt["suffix"] not in filename or ".swp" in filename:
                continue
            if not any(filename.startswith(d) for d in opt['datasets']):
                print(f"Skipping due to --datasets filter: {filename}")
                continue
            out_p = (
                opt['output_directory']
                + opt["model_name"].split('/')[-1]
                + f"/scores_{filename.split('.')[0]}.tsv"
            )
            if os.path.exists(out_p):
                print(f"Score file for {filename} already exists. Skipping.")
            else:
                print(f"Evaluating {os.path.join(root, filename)}")
                evaluator.update_evaluator(os.path.join(root, filename))
                score_types = (
                    REASONING_SCORES
                    if "esnli" in filename or "gsm8k" in filename
                    else UNSUPERVISED_SCORES
                )
                score_types = [st for st in score_types if st in opt['scores']]
                scores = evaluator.evaluate(score_types=score_types)
                save_scores(scores, out_p)
                print_and_reset_max_gpu_memory()
