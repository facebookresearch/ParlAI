#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import os
from typing import Dict, List
import pandas
import json
from projects.roscoe.utils import (
    split_gsm8k_gpt3_generations_to_steps,  # to normalize same as in annotations
)
from projects.roscoe.synthetic_evaluation.synthetic_roscoe import (
    SyntheticChain,
)
from projects.roscoe.baselines.constants import (
    Example,
    UseRef,
    BASELINE_SCORES,
    INPUT_DATA_SYNTHETIC,
    INPUT_DATA_FILES_HUMAN,
    INPUT_DATA_HUMAN,
    ALL_DATASETS,
    PERTURBATIONS,
)
from projects.roscoe.baselines.scores import (
    SCORES_TO_CLASS,
)
from nltk import sent_tokenize

################# Default settings
OUT_PATH = "./projects/roscoe/baseline_scores/"
SYNTHETIC_PATH = "./projects/roscoe/roscoe_data/synthetic_50%/"


class BaselineDataLoader:
    @classmethod
    def load_human(cls, dataset) -> List[Example]:
        with open(INPUT_DATA_FILES_HUMAN[dataset]) as f:
            raw = f.readlines()

        def parse_to_example(line):
            def normalize_SyntheticChain(s):
                return " ".join(sent_tokenize(s))

            def normalize_ParsedChain(s):
                return " ".join(s)

            jline = json.loads(line)
            r_chain = None
            # following kept in sync with `real_scorer.py`
            # We use the normalize functions as described above since baselines do not operate on steps, but we need to take
            # into account any characters that get changed when `real_scorer.py` parses things into steps.
            # Conflusingly, `real_scorer.py` uses "SyntheticChain" when it's real data...
            h_chain = normalize_SyntheticChain(jline["gpt-3"])
            context = normalize_SyntheticChain(
                jline["premise"] + " " + jline["hypothesis"]
            )
            if "gsm8k" in dataset:
                context = normalize_ParsedChain(sent_tokenize(jline["premise"]))
                h_chain = normalize_ParsedChain(
                    split_gsm8k_gpt3_generations_to_steps(jline["gpt-3"])
                )
                r_chain = normalize_ParsedChain(
                    jline["hypothesis"]
                    .split("IGNORE THIS. Ground truth here for reference. ")[1]
                    .split('\n')
                )
            if "esnli" in dataset:
                r_chain = normalize_SyntheticChain(
                    jline["explanation_1"]
                    + " "
                    + jline["explanation_2"]
                    + " "
                    + jline["explanation_3"]
                )
            return Example(context, h_chain, r_chain)

        return [parse_to_example(x) for x in raw]

    @classmethod
    def load_synthetic_datas(cls, dataset) -> Dict[str, List[Example]]:
        dataset_path_name = dataset
        if dataset == "math":
            dataset_path_name = "math_dataset"

        def parse_to_example(
            line,
        ):
            jline = json.loads(line)
            # This should be aligned with what is in synthetic_scorer.py
            h_chain = SyntheticChain(line=jline["dialog"][0][0]["steps"])
            r_chain = SyntheticChain(line=jline["dialog"][0][0]["original_steps"])
            context = SyntheticChain(line=jline["dialog"][0][0]["question"].split(". "))
            return Example(
                " ".join(context.chain),
                " ".join(h_chain.chain),
                " ".join(r_chain.chain),
            )

        result = {}
        for f_name in glob.glob(
            SYNTHETIC_PATH + f"{dataset_path_name}_synthetic/50*test.jsonl"
        ):
            seen_ps = 0
            for p in PERTURBATIONS:
                if p in f_name:
                    seen_ps += 1
            if seen_ps != 1:
                continue
            print("Synthetic data filename:", f_name)
            with open(f_name) as f:
                raw = f.readlines()
            result[
                dataset + "_" + f_name.split("/")[-1].replace(".jsonl", "_scores")
            ] = [parse_to_example(x) for x in raw]
        return result

    @classmethod
    def load_data(cls, dataset) -> Dict[str, List[Example]]:
        """
        Given a short name for the dataset, return a map that has descriptor for what
        the data from the dataset is, plus the examples (useful since synthetic data has
        different folders for Perturbations)
        """
        if dataset == "test":
            return {"test": [Example(x, x, x + 2) for x in range(8)]}
        if dataset in INPUT_DATA_HUMAN:
            return {dataset: cls.load_human(dataset)}
        assert dataset in INPUT_DATA_SYNTHETIC
        return cls.load_synthetic_datas(dataset)


def main(args):
    def make_path(p):
        import pathlib  # make sure out path exists

        pathlib.Path(os.path.dirname(p)).mkdir(parents=True, exist_ok=True)

    def save_scores_map(path, metric_to_score):
        print('saving scores to', path)
        datas = pandas.DataFrame.from_dict(metric_to_score)
        ordered = [x for x in BASELINE_SCORES if x in list(metric_to_score.keys())]
        datas = datas.reindex(columns=ordered)
        make_path(path)
        with open(path, "w+") as f:
            datas.to_csv(f)

    print("========================================= Done with module includes")
    datasets = args.dataset
    if "test" in datasets:
        datasets = ["test"]
    elif "human" in datasets:
        datasets = INPUT_DATA_FILES_HUMAN
    elif "synthetic" in datasets:
        datasets = INPUT_DATA_SYNTHETIC
    print("Getting scores for datasets", datasets)
    print("On scores", args.score)

    scorers_classes = set([SCORES_TO_CLASS[x] for x in args.score])
    scorers = [x() for x in scorers_classes]
    score_path_name = "-".join(args.score)

    use_ref = [
        x for x in UseRef if x.value in args.use_ref
    ]  # string -> enum conversion

    print("Using reference types", use_ref)

    print(
        "========================================= Done loading scorers; iterating through datasets"
    )

    for dataset in datasets:
        print("Scoring dataset", dataset)
        datas = BaselineDataLoader.load_data(dataset)
        print("Data loaded, about to start scoring")
        for out_name, examples in datas.items():
            with_ref_scores = {}
            no_ref_scores = {}
            for scorer in scorers:
                scores_raw = scorer.score_data(examples, use_ref)
                if UseRef.YES in scores_raw:
                    for metric, scores in scores_raw[UseRef.YES].items():
                        if len(scores) > 0:
                            with_ref_scores[metric] = scores
                        else:
                            print("with ref length of 0:", dataset, out_name, metric)
                if UseRef.NO in scores_raw:
                    for metric, scores in scores_raw[UseRef.NO].items():
                        if len(scores) > 0:
                            no_ref_scores[metric] = scores
                        else:
                            print("no ref length of 0:", dataset, out_name, metric)
            # Now we've got all our scores, save them
            out_for_dataset = os.path.join(args.out_dir, dataset, score_path_name)
            make_path(out_for_dataset)

            if len(with_ref_scores) > 0:
                save_scores_map(
                    os.path.join(out_for_dataset, f"{out_name}-with_ref.csv"),
                    with_ref_scores,
                )

            if len(no_ref_scores) > 0:
                save_scores_map(
                    os.path.join(out_for_dataset, f"{out_name}-no_ref.csv"),
                    no_ref_scores,
                )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        choices=ALL_DATASETS + ["test", "human", "synthetic"],
        nargs="+",
        default=ALL_DATASETS,
        help='name of datasets to score. If "test", "human", or "synthetic" used, will replace entire input with those.',
    )
    parser.add_argument(
        '--score',
        type=str,
        choices=list(SCORES_TO_CLASS.keys()),
        nargs="+",
        default=list(SCORES_TO_CLASS.keys()),
        help='name of scores to gen',
    )
    parser.add_argument(
        '--use-ref',
        type=str,
        choices=[x.value for x in UseRef],
        nargs="+",
        default=UseRef.NO.value,
        help='do we want to generate reference-based or reference-free scores',
    )
    parser.add_argument(
        '--out-dir', type=str, help='output directory', default=OUT_PATH
    )

    args = parser.parse_args()
    main(args)
