#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Helper function to run correlation analysis on synthetic data evaluations, scattered
across model-specific folders.

Usage:
python projects/roscoe/meta_evaluation/roscoe_synthetic_correlations.py --dataset-name aqua
"""

import csv
from glob import glob
import json
import os
import pathlib
import numpy as np

import pandas
from parlai.core.params import ParlaiParser
from projects.roscoe.meta_evaluation.correlations import Analyzer
from projects.roscoe.score import (
    EMB_MODEL_SCORES,
    EMBEDDING_MODEL_NAMES,
    GRAMMAR_MODEL_SCORES,
    LANGUAGE_MODEL_SCORES,
    NLI_MODEL_SCORES,
    REASONING_SCORES,
    SCORE_GROUPS,
    SUPERVISED_SCORES,
    UNSUPERVISED_SCORES,
)
from typing import Callable, List

DEFAULT_FILE_PATH = "./projects/roscoe/roscoe_data/synthetic_50%/"
DEFAULT_SCORE_PATH = "./projects/roscoe/scores/"
DEFAULT_OUTPUT_PATH = "./projects/roscoe/correlations/"

PERTURBATION_COLUMN = "perturbed"


def is_sentinel(file_name: str) -> bool:
    names = file_name.split("_")
    return len(names) > 3 and names[0][0].isdigit()


def is_single_perturbation(file_name: str) -> bool:
    names = file_name.split("_")
    return len(names) == 3 and names[0][0].isdigit()


def fetch_to_df(file_name) -> pandas.DataFrame:
    df = pandas.read_csv(file_name, delimiter=r"\s+")
    return df


def fetch_parturbation_mark(
    scores: pandas.DataFrame, path_to_data: str
) -> pandas.DataFrame:
    with open(path_to_data) as data_file:
        for idx, line in enumerate(data_file.readlines()):
            blob = json.loads(line)
            if idx in scores.index:
                scores.loc[idx, PERTURBATION_COLUMN] = int(
                    not blob[PERTURBATION_COLUMN]
                )
    assert not any(scores.columns.duplicated())

    return scores


def rearrange_cols_and_save_to_csv(final_dataframe, out_f):
    col_names = final_dataframe.columns.tolist()
    col_names.remove("fold")
    col_names.remove("corr_type")
    col_names.remove("tune_type")
    col_names.remove("ID")

    # filter for scores that appear in the data
    no_ref_metrics = [s for s in UNSUPERVISED_SCORES if s in col_names]
    ref_metrics = [s for s in SUPERVISED_SCORES if s in col_names]

    for score in no_ref_metrics + ref_metrics:
        col_names.remove(score)

    rearranged = (
        ["fold", "corr_type", "tune_type"]
        + no_ref_metrics
        + sorted(col_names)
        + ref_metrics
    )
    final_dataframe = final_dataframe[rearranged]

    final_dataframe.to_csv(out_f)

    print("Final results path is ", out_f)


def unique_subdirectories(
    root_directory: str,
    filter_fn: Callable[[str], bool],
) -> List[str]:
    """
    Given two directories with common subdirectory names:

    /path/to/first/
        model_name/
    /path/to/second/
        model_name/

    Returns list of unique 'model_name'
    """
    subdirs = []
    for sub in os.listdir(root_directory):
        if not os.path.isdir(os.path.join(root_directory, sub)):
            continue
        if filter_fn(sub):
            continue
        subdirs.append(sub)
    return subdirs


def is_language(s_name, m_name):
    if s_name in NLI_MODEL_SCORES and m_name == 'inference':
        return True
    return False


def is_inference(s_name, m_name):
    if s_name in GRAMMAR_MODEL_SCORES + LANGUAGE_MODEL_SCORES and m_name == 'language':
        return True
    return False


def is_semantic(s_name, m_name):
    if s_name in EMB_MODEL_SCORES and m_name != 'inference' and m_name != 'language':
        return True
    return False


def not_found(s_name, m_name):
    return (
        is_language(s_name, m_name)
        or is_inference(s_name, m_name)
        or is_semantic(s_name, m_name)
    )


def get_summary(
    model_names: List[str],
    final_out: str,
    summary_out: str,
    stat_name: str,
    pval_name: str,
):
    """
    For each embedding model, get average and max correlation between score and
    perturbation ground truth.
    """
    # The final summary file
    summary_lines = []
    metrics = []
    scores = {}
    n_meta_columns = 4
    with open(final_out) as file_obj:
        reader_obj = csv.reader(file_obj)

        r = 0
        for row in reader_obj:
            if r == 0:
                metrics = row[n_meta_columns:]
                r += 1
                continue

            data_name = row[0]
            data_split = row[1]  # train/dev/test
            correlation_type = row[2]
            model = row[3]

            if model not in scores:
                scores[model] = {}
            if correlation_type not in scores[model]:
                scores[model][correlation_type] = {}
                for metric in metrics:
                    scores[model][correlation_type][metric] = []

            if not is_single_perturbation(data_name) and not is_sentinel(data_name):
                continue

            for metric_idx in range(len(metrics)):
                score_value = row[n_meta_columns + metric_idx]
                if score_value != '':
                    scores[model][correlation_type][metrics[metric_idx]].append(
                        float(score_value)
                    )
                elif not_found(metrics[metric_idx], model):
                    print(
                        f"Missing {data_split} {metrics[metric_idx]} value for {model}"
                    )

        ### Report AVERAGE

        for model, score_dict in scores.items():
            if model not in model_names:
                continue
            summary_lines.append(
                ','.join(
                    [
                        model,
                        f'Average {stat_name}',
                        f'Average {pval_name}',
                        f'Max {stat_name}',
                        f'{pval_name}',
                    ]
                )
                + "\n"
            )
            stat_score = score_dict[stat_name]
            stat_pvalue = score_dict[pval_name]
            for metric in stat_score.keys():
                if metric not in REASONING_SCORES:
                    print(f"{metric} not in known scores")
                    continue
                if len(stat_score[metric]) == 0:
                    if not_found(metric, model):
                        print(f"No {metric} scores found for {model}")
                    continue

                average_score = np.average(stat_score[metric])
                average_p_value = np.average(stat_pvalue[metric])

                max_score = np.max(stat_score[metric])
                max_pvalue = stat_pvalue[metric][stat_score[metric].index(max_score)]
                summary_lines.append(
                    ','.join(
                        [
                            metric,
                            str(average_score),
                            str(average_p_value),
                            str(max_score),
                            str(max_pvalue),
                        ]
                    )
                    + "\n"
                )
            summary_lines.append("\n")

    with open(summary_out, "w") as f:
        f.writelines(summary_lines)

    print(f"summary written to {summary_out}")


def get_correlations(
    dataset_name: str,
    data_path: str,
    scores_path: str,
    out_path: str,
    sentinel: bool,
):
    score_file_prefix = f"scores_{dataset_name}_synthetic_"
    stats_output = os.path.join(out_path, f"{dataset_name}_statistics/")
    final_output_file = os.path.join(out_path, f"final/{dataset_name}.csv")
    summary_output_file = os.path.join(out_path, f"final/{dataset_name}_summary.csv")
    base_df_output = os.path.join(stats_output, "base_df")
    correlations = Analyzer(reference_column=PERTURBATION_COLUMN)

    os.makedirs(stats_output, exist_ok=True)
    os.makedirs(os.path.dirname(final_output_file), exist_ok=True)
    os.makedirs(base_df_output, exist_ok=True)

    ### Grab all of scores and dump them into one big pandas file
    for file_name in glob(scores_path + "/**/" + score_file_prefix + "*"):
        tune_type = file_name.split("/")[-2]

        perturbation_name = file_name.split("/")[-1][
            len(score_file_prefix) : -len(".tsv")
        ]

        names = perturbation_name.split("_")

        print(f"Reading scores: {tune_type} {perturbation_name}")

        if is_single_perturbation(perturbation_name) and not sentinel:
            percent = int(names[0][:-1])
            fold = names[-1]
            data_name = f"{percent}%_{names[1]}_{fold}"
        elif is_sentinel(perturbation_name) and sentinel:
            percent = int(names[0][:-1])
            fold = names[-1]
            data_name = f"{percent}%_{'_'.join(names[1:-1])}_{fold}"
        else:
            print(f"Skipping {file_name}")
            continue

        df = fetch_to_df(file_name)

        valid_data_file = os.path.join(
            data_path, dataset_name + "_synthetic", data_name + ".jsonl"
        )

        if not os.path.isfile(valid_data_file):
            print(f"{data_name}.jsonl not found. Skipping.")
            continue

        df = fetch_parturbation_mark(scores=df, path_to_data=valid_data_file)

        df.to_csv(os.path.join(base_df_output, perturbation_name + ".csv"))

        stats = correlations.get_statistics(df)

        for corr in [correlations.statistic, correlations.pval]:
            s = stats[corr]

            # needs to be synced with glob for merging to final below
            output_path = os.path.join(
                stats_output,
                dataset_name,
                tune_type,
                corr,
                perturbation_name + ".csv",
            )
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            s[[PERTURBATION_COLUMN]].to_csv(output_path)

    # Get the stats into one table
    results_list = []

    # Needs to be kept in sync w/ output_path above
    print("Getting summary")
    for file_name in glob(os.path.join(stats_output, dataset_name, "*/*/*.csv")):
        df = pandas.read_csv(file_name)
        df = df.transpose()
        df.columns = df.iloc[0]
        df.drop(df.index[0], inplace=True)
        df.index = [file_name.split("/")[-1][: -len(".csv")]]
        df["fold"] = pathlib.Path(file_name).stem.split("_")[-1]
        df["corr_type"] = file_name.split("/")[-2]
        df["tune_type"] = file_name.split("/")[-3]
        results_list.append(df)
    res = pandas.concat(results_list)
    rearrange_cols_and_save_to_csv(final_dataframe=res, out_f=final_output_file)

    subdirs = unique_subdirectories(
        scores_path,
        filter_fn=lambda d: d not in EMBEDDING_MODEL_NAMES + SCORE_GROUPS,
    )
    get_summary(
        model_names=subdirs,
        final_out=final_output_file,
        summary_out=summary_output_file,
        stat_name=correlations.statistic,
        pval_name=correlations.pval,
    )


if __name__ == '__main__':
    parser = ParlaiParser(add_model_args=True)
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Name of dataset e.g. entailment_bank. Will be used to name saved files.',
    )
    parser.add_argument(
        '--data-file-path',
        type=str,
        default=DEFAULT_FILE_PATH,
        help='Absolute path to directory where data files are.',
    )
    parser.add_argument(
        '--scores-path',
        type=str,
        default=DEFAULT_SCORE_PATH,
        help=(
            'Absolute path to directory where score tsv files are. Each subdirectory is '
            'expected to be the name of the text embedding model used.'
        ),
    )
    parser.add_argument(
        '--out-path',
        type=str,
        default=DEFAULT_OUTPUT_PATH,
        help='Absolute path to directory where the script will save the analysis output.',
    )
    parser.add_argument(
        '--sentinel-scores',
        type=bool,
        default=False,
        help='If sentinel scores are to be analysed',
    )
    opt = parser.parse_args()

    get_correlations(
        dataset_name=opt['dataset_name'],
        data_path=opt['data_file_path'],
        scores_path=opt['scores_path'],
        out_path=opt['out_path'],
        sentinel=opt['sentinel_scores'],
    )
