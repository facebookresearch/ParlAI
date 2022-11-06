#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Helper function to run correlation analysis on annotated data evaluations, scattered
across model-specific folders.

Usage:
python projects/roscoe/meta_evaluation/roscoe_correlations.py --dataset drop --out-dir /path/to/your/orrelations/
"""
import argparse
from collections import defaultdict
import csv
import math
import os
import pathlib
from typing import Dict, Iterable, List, Optional, Set, Tuple, Union

import pandas
from projects.roscoe.meta_evaluation.correlations import Analyzer

from projects.roscoe.score import (
    ROSCOE_SA,
    ROSCOE_SS,
    ROSCOE_LI,
    ROSCOE_LC,
    SUPERVISED_SCORES,
    UNSUPERVISED_SCORES,
)
from projects.roscoe.utils import (
    assert_all_elements_same_length,
    ordered_union,
    split_list,
)
from projects.roscoe.meta_evaluation.table_file_writing import (
    ANNOTATION_COLUMN_ORDER,
    COLUMN_RENAMES,
    MODELS,
    ROSCOE_SECTIONS,
    STEP_COLUMN_RENAMES,
    section_rows,
    write_every_dataset_summary_table_tex,
    write_every_dataset_final_summary_table_tex,
    write_granular_summary_tex,
)

DEFAULT_FILE_PATH = "./projects/roscoe/roscoe_data/annotated/"
DEFAULT_SCORE_PATH = "./projects/roscoe/scores/"
DEFAULT_BASELINE_PATH = "./projects/roscoe/baseline_scores/"
DEFAULT_OUTPUT_PATH = "./projects/roscoe/correlations/"


REPORTED_BASELINES = [
    "rouge_l",
    "bleurt",
    "bertScore_f",
    "bartScore_f",
    "bartscore_cnn_para_f",
    "bartscore_finetuned_f",
    "prism_avg",
    "ctc_relevance_summary",
    "ctc_consistency_summary",
]
EXTRA_REPORTED_BASELINES = [
    "rouge_1",
    "rouge_2",
]


POSITIVE_ANNOTATIONS = {
    'full_consistentContext',
    'full_coherent',
    'full_consistentSelf',
    'step_questions_useful',
}


NEGATIVE_ANNOTATIONS = {
    'full_missingStep',
    'full_newMissingStep',
    'full_newClearContradiction',
    'step_questions_contradictoryToRelationship',
    'step_questions_irrelevant_',
    'step_questions_commonSense',
    'step_questions_contradictoryToSteps',
    'step_questions_logicMathError',
    'step_questions_internalHallucination',
    'step_questions_externalHallucination',
    'step_questions_redundant_',
    'step_questions_tenseError',
    'step_questions_otherGrammarError',
    'step_questions_other_',
    'step_questions_newGrammar',
    'step_questions_newContradictContext',
    'step_questions_newLogicalDeduction',
    'step_questions_newFinalAnswerWrong',
    'step_questions_newExtraUselessInfo',
    'step_questions_newIntermediateFactualInfo',
    'step_questions_newDroppableStep',
    'step_questions_newWorldKnowledge',
    'step_questions_newMathError',
}


DATASET_TO_CONFIG = {
    "drop": {
        "human_labels": {
            "labels": f"{DEFAULT_FILE_PATH}/drop.csv",
            "skip_indices": [],
        },
        "baseline_scores": f"{DEFAULT_BASELINE_PATH}/drop_test_merged_scores.csv",
        "alignment_scores": [
            os.path.join(
                DEFAULT_SCORE_PATH,
                m,
                "scores_drop.tsv",
            )
            for m in MODELS
        ],
    },
    "gsm8k": {
        "human_labels": {
            "labels": f"{DEFAULT_FILE_PATH}/gsm8k.csv",
            "skip_indices": [],
        },
        "baseline_scores": f"{DEFAULT_BASELINE_PATH}/gsm8k_test_merged_scores.csv",
        "alignment_scores": [
            os.path.join(
                DEFAULT_SCORE_PATH,
                m,
                "scores_gsm8k.tsv",
            )
            for m in MODELS
        ],
    },
    "esnli": {
        "human_labels": {
            "labels": f"{DEFAULT_FILE_PATH}/esnli.csv",
            "skip_indices": [],
        },
        "baseline_scores": f"{DEFAULT_BASELINE_PATH}/esnli_test_merged_scores.csv",
        "alignment_scores": [
            os.path.join(
                DEFAULT_SCORE_PATH,
                m,
                "scores_esnli.tsv",
            )
            for m in MODELS
        ],
    },
    "cosmos": {
        "human_labels": {
            "labels": f"{DEFAULT_FILE_PATH}/cosmos.csv",
            "skip_indices": [
                '145',
                '147',
                '150',
                '151',
                '155',
                '156',
                '162',
                '164',
                '169',
                '178',
                '182',
                '191',
                '192',
                '193',
                '194',
                '195',
            ],
        },
        "baseline_scores": f"{DEFAULT_BASELINE_PATH}/cosmos_test_merged_scores.csv",
        "alignment_scores": [
            os.path.join(
                DEFAULT_SCORE_PATH,
                m,
                "scores_cosmos.tsv",
            )
            for m in MODELS
        ],
    },
    "semeval": {
        "human_labels": {
            "labels": f"{DEFAULT_FILE_PATH}/semeval.csv",
            "skip_indices": [
                '8',
                '11',
                '12',
                '25',
                '29',
                '42',
                '45',
                '48',
                '49',
                '52',
                '95',
                '199',
            ],
        },
        "baseline_scores": f"{DEFAULT_BASELINE_PATH}/semeval_test_merged_scores.csv",
        "alignment_scores": [
            os.path.join(
                DEFAULT_SCORE_PATH,
                m,
                "scores_semevalcommonsense.tsv",
            )
            for m in MODELS
        ],
    },
}


def rename_columns_for_display(column_row: List[str]) -> List[str]:
    return [rename_column_title(t) for t in column_row]


def rename_column_title(title: str) -> str:
    if title.endswith('_result'):
        title = title[: -len('_result')]
    if title.startswith('0_full_'):
        title = title[len('0_full_') :]
    title = title.replace('_questions_', '_')
    title = title.replace('step_step_', '')
    title = title.replace('commonSense', 'factual')
    return title


def read_space_separated_to_df(file_name):
    return pandas.read_csv(file_name, delimiter=r"\s+")


def parse_csv_with_idx(fn):
    rows = {}
    tags = []
    with open(fn) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                tags = row
            else:
                escaped_row = [r.encode("unicode_escape").decode("utf-8") for r in row]
                row_i = {}
                for r in range(len(escaped_row)):
                    row_i[tags[r]] = escaped_row[r]
                rows[row_i["metadata_example_idx"]] = row_i
            line += 1
    return rows, tags


def parse_scorers_baseline(fn, rows):
    tsvtags = []
    newrows = {}
    with open(fn) as tsv_file:
        csv_reader = csv.reader(tsv_file, delimiter=',')
        line = 0
        for row in csv_reader:
            if line == 0:
                tsvtags = row
            else:
                if str(line - 1) in rows:
                    newrows[str(line - 1)] = rows[str(line - 1)]
                    for r in range(len(row)):
                        newrows[str(line - 1)][tsvtags[r]] = row[r]
            line += 1
    return newrows, tsvtags


def parse_alignment_scorers(fn, rows, tags):
    newrows = {}
    df = read_space_separated_to_df(fn)

    tsvtags = list(df.loc[0].keys())
    data_dict = df.to_dict()

    for key, value in data_dict.items():
        for line in range(len(value)):
            if str(line) not in rows:
                continue
            newrows[str(line)] = rows[str(line)]
            newrows[str(line)][key] = str(value[line])
    # dedupe tags
    new_tags = tags + [t for t in tsvtags if t not in tags]
    return newrows, new_tags


def clean_tags(tags, reference_option: str):
    remove_list = ["Line #", "is_Perturbed", "GoldScore", "ID", ""]
    # outdated scores
    remove_list += ["coherence_max", "coherence_step"]
    # unused baselines
    remove_list += [
        "BARTScore Informativeness",
        "BARTScore Recall",
        "BERTScore_P",
        "BERTScore_R",
        "bart_score_para_hypo_ref",
        "bart_score_para_ref_hypo",
        "bart_score_para_harm_f",
        "prism_ref_hypo",
        "prism_hypo_ref",
        "Finetuned-BARTScore Recall",
        "Finetuned-BARTScore Informativeness",
    ]
    # with or without reference reasoning chain
    if reference_option == 'yes':
        remove_list += UNSUPERVISED_SCORES
    elif reference_option == 'no':
        remove_list += SUPERVISED_SCORES
    return [t for t in tags if t not in remove_list]


def step_numbers_from_step_labels(stepwise_labels: Iterable[str]) -> List[int]:
    return sorted(set(step_number_from_label(h) for h in stepwise_labels))


def step_number_from_label(label: str) -> int:
    for component in label.split('_'):
        if component.isdigit():
            return int(component)
    raise Exception(f"No step number found for label: {label}")


def step_number_from_score(score: str) -> int:
    return int(score.split('_')[1])


def root_score_from_stepwise_score(score_name: str) -> str:
    return '_'.join(score_name.split('_')[2:])


def root_label_from_numbered_step_label(label_name: str) -> str:
    if label_name.split('_')[0].isdigit():
        return '_'.join(label_name.split('_')[1:])
    else:
        return '_'.join(label_name.split('_')[2:])


def contains_number(title: str) -> bool:
    components = title.split('_')
    if len(components) == 1:
        return title.isdigit()
    return any(c.isdigit() for c in components)


def is_human_label_column_title(t: str) -> bool:
    # e.g. any_step_any_issue
    # note that the 'any_' prefix is a safe assumption for catching only human labels, since the concept of "any" doesn't
    # make sense with continuous scores
    is_aggregate_format = t.startswith('any_') or 'S-' in t or 'ANY_ERROR' in t
    if is_aggregate_format:
        return True
    # e.g. 0_full_overall, 1_step_step_questions_irrelevant
    is_original_format = (
        t.split('_')[0].isdigit() and ('_full_' in t or '_step_' in t)
    ) or t.endswith('_result')
    return is_original_format


def is_label_stepwise(label_title: str) -> bool:
    """
    Human annotations for which there is one label per example per step.

    These will numbered or be labeled 'last'.
    """
    if label_title.startswith('last_'):
        return True
    # e.g. 0_full_commonsenseSituationClaim
    components = label_title.split('_')
    if len(components) <= 1:
        return False
    return contains_number(label_title) and components[1] != 'full'


def is_label_stepwise_aggregate(label_title: str) -> bool:
    """
    Labels that are a concatenation of a particular stepwise human annotation at every
    step.
    """
    if contains_number(label_title):
        return False
    # e.g. step_questions_useful_result, middle_step_questions_tenseError_result, S-FACT
    labeled_as_such = (
        label_title.startswith('step_')
        or label_title.startswith('middle_')
        or label_title.startswith('S-')
    )
    return labeled_as_such


def is_score_stepwise(score_title: str) -> bool:
    """
    Scores for which there is one label per example per step.
    """
    if score_title.startswith('last_'):
        return True
    # special exceptions to "contains number" rule
    if any(score_title == s for s in ('rouge_1', 'rouge_2')):
        return False
    return contains_number(score_title)


def is_score_stepwise_aggregate(score_title: str) -> bool:
    """
    Scores that are a concatentation of a stepwise score across all steps.
    """
    if contains_number(score_title):
        return False
    labeled_as_such = score_title.startswith('stepwise_') or score_title.startswith(
        'middle_'
    )
    return labeled_as_such


def remove_empty_columns(values_by_title):
    def _is_column_empty(title: str) -> bool:
        return all(math.isnan(s) for s in values_by_title[title])

    empty_columns = [t for t in values_by_title.keys() if _is_column_empty(t)]
    for c in empty_columns:
        del values_by_title[c]

    return values_by_title


def filter_and_order_column_titles(
    human_labels: List[str],
    stepwise_labels: List[str],
) -> List[str]:
    # columns
    chain_label_names = [h for h in human_labels if h.startswith('0_full')]
    agg_stepwise_label_names = [
        s for s in stepwise_labels if is_label_stepwise_aggregate(s)
    ]
    stepwise_labels_not_numbered = [
        s for s in stepwise_labels if not contains_number(s)
    ]
    agg_stepwise_agg_label_names, agg_stepwise_raw_label_names = split_list(
        agg_stepwise_label_names,
        lambda n: 'S-' in n and not n.startswith('middle_'),
    )
    last_step_labels = [
        l for l in stepwise_labels_not_numbered if l.startswith('last_')
    ]
    # pull the "any step" aggregation labels to the front, with *_any_issue labels first
    agg_rating_labels = [
        l
        for l in human_labels
        if l.startswith('any_') and not l.startswith('any_middle_')
    ]
    top_agg_rating_labels = [
        'ANY_ERROR',
        'any_step_any_issue',
    ]
    agg_rating_labels = [l for l in agg_rating_labels if l not in top_agg_rating_labels]

    # then come the others
    non_agg_rating_labels = [
        l for l in stepwise_labels_not_numbered if l not in agg_rating_labels
    ]
    seq_aggregates = [
        'ANY_ERROR',
        '0_full_overall_result'
        if '0_full_overall_result' in human_labels
        else '0_full_newOverall_result',
        '0_full_coherent_result'
        if '0_full_coherent_result' in human_labels
        else '0_full_newCoherent_result',
    ]
    column_titles = ordered_union(
        [
            seq_aggregates,
            chain_label_names,
            top_agg_rating_labels,
            agg_stepwise_agg_label_names,
            last_step_labels,
            agg_rating_labels,
            agg_stepwise_label_names,
            non_agg_rating_labels,
        ]
    )
    return column_titles


def filter_and_order_row_titles(
    score_titles: List[str],
):
    r_sa = [s for s in score_titles if any(r in s for r in ROSCOE_SA)]
    r_ss = [s for s in score_titles if any(r in s for r in ROSCOE_SS)]
    r_li = [s for s in score_titles if any(r in s for r in ROSCOE_LI)]
    r_lc = [s for s in score_titles if any(r in s for r in ROSCOE_LC)]
    roscoe_scores = r_sa + r_ss + r_li + r_lc
    other_baselines = [s for s in score_titles if s not in roscoe_scores]
    row_titles = ordered_union(
        [REPORTED_BASELINES, r_sa, r_ss, r_li, r_lc, other_baselines]
    )
    return row_titles


def make_correlations_table(values_by_title: Dict[str, List[float]]):
    """
    Given all the annotations and scores, finds the correlations between comparable
    pairs.
    """
    # separate scores and labels
    human_column_titles, score_row_titles = split_list(
        sorted(values_by_title.keys()),
        is_human_label_column_title,
    )

    # separate out step-wise labels
    stepwise_labels = [h for h in human_column_titles if is_label_stepwise(h)]

    # final row & column ordering
    column_titles = filter_and_order_column_titles(
        human_labels=human_column_titles,
        stepwise_labels=stepwise_labels,
    )
    row_titles = filter_and_order_row_titles(score_titles=score_row_titles)

    # this is the top row, where the upper-left-most cell is the title of the table
    somersd_table = [["SomersD"] + column_titles]
    pval_table = [["p"] + column_titles]

    for score_title in row_titles:
        somersd_row = [score_title]
        pval_row = [score_title]
        for i, human_title in enumerate(column_titles):
            assert (
                human_title == somersd_table[0][i + 1]
            ), "values don't align with columns!"

            if len(values_by_title[human_title]) != len(values_by_title[score_title]):
                raise Exception(
                    f"Attempting to compare values whose rows do not align. Score: {score_title}, Label: {human_title}"
                )

            analyzer = Analyzer(reference_column=human_title)

            somersd_stat = analyzer.correlation_stat(
                df=pandas.DataFrame.from_dict(
                    {
                        human_title: values_by_title[human_title],
                        score_title: values_by_title[score_title],
                    }
                ),
                hypo_column=score_title,
            )

            correl = somersd_stat.statistic
            p_val = somersd_stat.pvalue

            somersd_row.append(f"{correl:.3f}")
            pval_row.append(p_val)
        somersd_table.append(somersd_row)
        pval_table.append(pval_row)

    return somersd_table, pval_table


def generate_every_dataset_summary_table(
    summaries_by_dataset: Dict[
        str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]
    ],
    summary_column: str,
    out_dir: str,
):
    out_path = os.path.join(out_dir, "seq_summary_granular.csv")
    with open(out_path, 'w') as f:
        over_datasets = pandas.concat(
            [
                sections['baselines'].rename(columns={summary_column: name})
                for name, sections in summaries_by_dataset.items()
                if sections is not None
            ],
            axis=1,
        )
        f.write(over_datasets.to_csv())

        for section in ('ROSCOE-SA', 'ROSCOE-SS', 'ROSCOE-LI', 'ROSCOE-LC'):
            if section in {'ROSCOE-SA', 'ROSCOE-SS'}:
                for model in MODELS:
                    over_datasets = pandas.concat(
                        [
                            sections[section][model]
                            .loc[ROSCOE_SECTIONS[section]]
                            .rename(columns={summary_column: name})
                            for name, sections in summaries_by_dataset.items()
                            if sections is not None
                        ],
                        axis=1,
                    )
                    f.write(f"{section} - {model}\n")
                    f.write(over_datasets.to_csv(header=False))
            else:
                over_datasets = pandas.concat(
                    [
                        sections[section]
                        .loc[ROSCOE_SECTIONS[section]]
                        .rename(columns={summary_column: name})
                        for name, sections in summaries_by_dataset.items()
                        if sections is not None
                    ],
                    axis=1,
                )
                f.write(f"{section}\n")
                f.write(over_datasets.to_csv(header=False))

    print(f"Summary written to: {out_path}")


def generate_every_dataset_final_summary_table(
    summaries_by_dataset: Dict[str, pandas.DataFrame],
    summary_column: str,
    out_dir: str,
):
    summary_all_datasets = pandas.concat(
        [
            df.rename(columns={summary_column: name})
            for name, df in summaries_by_dataset.items()
            if df is not None
        ],
        axis=1,
    )

    summary_out_path = os.path.join(out_dir, "seq_summary.csv")
    summary_all_datasets.to_csv(summary_out_path)
    print(f"Summary written to: {summary_out_path}")

    return summary_all_datasets


def granular_summary(
    correls_by_model: Dict[str, pandas.DataFrame],
    pvals_by_model: Dict[str, pandas.DataFrame],
):
    columns = [
        "newOverall",
        "newCoherent",
        "any_newExtraUselessInfo",
        "any_newIntermediateFactualInfo",
        "any_newDroppableStep",
        "newMissingStep",
        "any_newContradictContext",
        "any_newWorldKnowledge",
        "any_newLogicalDeduction",
        "any_newMathError",
        "any_newGrammar",
    ]

    summary_sections = {}
    pval_sections = {}
    for model, correls_df in correls_by_model.items():
        seq_lvl_columns = correls_df[columns]
        pval_seq_lvl_columns = pvals_by_model[model][columns]

        baseline_score_names = EXTRA_REPORTED_BASELINES + REPORTED_BASELINES
        summary_sections['baselines'] = seq_lvl_columns.loc[
            baseline_score_names
        ].astype(float)
        pval_sections['baselines'] = pval_seq_lvl_columns.loc[baseline_score_names]

        data = []
        row_titles = []
        for agg_score, group_scores in ROSCOE_SECTIONS.items():
            # some models don't have all scores; don't use those to aggregate
            if not all(g in seq_lvl_columns[c] for g in group_scores for c in columns):
                continue

            group_scores_df = seq_lvl_columns.loc[group_scores].astype(float)
            sig_group_scores_df = group_scores_df[
                pval_seq_lvl_columns.loc[group_scores] < 0.05
            ]

            # get aggregate score correlation per annotation
            group_summary_correlations = sig_group_scores_df.max().tolist()
            data.append(group_summary_correlations)
            row_titles.append(agg_score)

            for score in group_scores:
                if score in row_titles:
                    continue
                data.append(group_scores_df.loc[score].tolist())
                row_titles.append(score)

        model_df = pandas.DataFrame(
            data=data,
            index=row_titles,
            columns=columns,
        )

        for emb_section in ('ROSCOE-SA', 'ROSCOE-SS'):
            section_df = model_df.loc[section_rows(emb_section)]
            if emb_section not in summary_sections:
                summary_sections[emb_section] = {}
                pval_sections[emb_section] = {}
            summary_sections[emb_section][model] = section_df
            pval_sections[emb_section][model] = pval_seq_lvl_columns.loc[
                ROSCOE_SECTIONS[emb_section]
            ]

        summary_sections['ROSCOE-LI'] = model_df.loc[section_rows('ROSCOE-LI')]
        pval_sections['ROSCOE-LI'] = pval_seq_lvl_columns.loc[
            ROSCOE_SECTIONS['ROSCOE-LI']
        ]
        summary_sections['ROSCOE-LC'] = model_df.loc[section_rows('ROSCOE-LC')]
        pval_sections['ROSCOE-LC'] = pval_seq_lvl_columns.loc[
            ROSCOE_SECTIONS['ROSCOE-LC']
        ]

    return summary_sections, pval_sections


def write_granular_summary(
    dataset_name: str,
    granular_sections: Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]],
    out_dir: str,
):
    def _format_section(df):
        return df.rename(columns=COLUMN_RENAMES)[ANNOTATION_COLUMN_ORDER]

    out_path = os.path.join(out_dir, f"{dataset_name}_summary_granular.csv")
    with open(out_path, 'w') as f:
        # baselines
        formatted_section = _format_section(granular_sections['baselines'])
        f.write(formatted_section.to_csv())
        # ROSCOE scores, by section
        for section in ('ROSCOE-SA', 'ROSCOE-SS', 'ROSCOE-LI', 'ROSCOE-LC'):
            if section in {'ROSCOE-SA', 'ROSCOE-SS'}:
                for model in MODELS:
                    formatted_section = _format_section(
                        granular_sections[section][model].loc[ROSCOE_SECTIONS[section]]
                    )
                    f.write(f"{section} - {model}\n")
                    f.write(formatted_section.to_csv(header=False))
            else:
                formatted_section = _format_section(
                    granular_sections[section].loc[ROSCOE_SECTIONS[section]]
                )
                f.write(f"{section}\n")
                f.write(formatted_section.to_csv(header=False))

    print(f"Granular summary of {dataset_name} written to: {out_path}")


def single_dataset_summary(
    correls_by_model: Dict[str, pandas.DataFrame],
) -> Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]]:
    summary_column = 'ANY_ERROR'

    summary_sections = {}
    for model, correls_df in correls_by_model.items():
        seq_lvl_columns = correls_df[[summary_column]]

        baseline_score_names = EXTRA_REPORTED_BASELINES + REPORTED_BASELINES
        summary_sections['baselines'] = seq_lvl_columns.loc[
            baseline_score_names
        ].astype(float)

        data = []
        row_titles = []
        for agg_score, group_scores in ROSCOE_SECTIONS.items():
            # some models don't have all scores; don't use those to aggregate
            if not all(g in seq_lvl_columns[summary_column] for g in group_scores):
                continue

            group_summary_correlation = max(
                float(seq_lvl_columns[summary_column][gs]) for gs in group_scores
            )
            data.append([group_summary_correlation])
            row_titles.append(agg_score)

            for score in group_scores:
                if score in row_titles:
                    continue
                data.append([float(seq_lvl_columns[summary_column][score])])
                row_titles.append(score)

        model_df = pandas.DataFrame(
            data=data,
            index=row_titles,
            columns=[summary_column],
        )

        for emb_section in ('ROSCOE-SA', 'ROSCOE-SS'):
            section_df = model_df.loc[section_rows(emb_section)]
            if emb_section not in summary_sections:
                summary_sections[emb_section] = {}
            summary_sections[emb_section][model] = section_df

        summary_sections['ROSCOE-LI'] = model_df.loc[section_rows('ROSCOE-LI')]
        summary_sections['ROSCOE-LC'] = model_df.loc[section_rows('ROSCOE-LC')]

    return summary_sections


def single_dataset_summary_max(
    correls_by_model: Dict[str, pandas.DataFrame],
    pvals_by_model: Dict[str, pandas.DataFrame],
) -> Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]]:

    columns = list(STEP_COLUMN_RENAMES.keys())

    def _format_section(df):
        return df.rename(columns=STEP_COLUMN_RENAMES)[ANNOTATION_COLUMN_ORDER[2:]]

    def _max_df(df):
        return pandas.DataFrame({'MAX': df.max(axis=1)})

    summary_sections = {}
    for model, correls_df in correls_by_model.items():

        seq_lvl_columns = _format_section(correls_df[columns])
        pval_reported_columns = _format_section(pvals_by_model[model][columns])

        baseline_score_names = EXTRA_REPORTED_BASELINES + REPORTED_BASELINES
        summary_sections['baselines'] = _max_df(
            seq_lvl_columns.loc[baseline_score_names].astype(float)[
                pval_reported_columns.loc[baseline_score_names] < 0.05
            ]
        )

        data = []
        row_titles = []
        for agg_score, group_scores in ROSCOE_SECTIONS.items():
            # some models don't have all scores; don't use those to aggregate
            if not all(
                g in seq_lvl_columns[c]
                for g in group_scores
                for c in ANNOTATION_COLUMN_ORDER[2:]
            ):
                continue

            sig_group_scores_df = seq_lvl_columns.loc[group_scores].astype(float)[
                pval_reported_columns.loc[group_scores] < 0.05
            ]
            group_summary_correlation = sig_group_scores_df.max().max()
            data.append([group_summary_correlation])
            row_titles.append(agg_score)

            for score in group_scores:
                if score in row_titles:
                    continue
                data.append([sig_group_scores_df.loc[[score]].max(axis=1).max()])
                row_titles.append(score)

        model_df = pandas.DataFrame(
            data=data,
            index=row_titles,
            columns=['MAX'],
        )

        for emb_section in ('ROSCOE-SA', 'ROSCOE-SS'):
            section_df = model_df.loc[section_rows(emb_section)]
            if emb_section not in summary_sections:
                summary_sections[emb_section] = {}
            summary_sections[emb_section][model] = section_df

        summary_sections['ROSCOE-LI'] = model_df.loc[section_rows('ROSCOE-LI')]
        summary_sections['ROSCOE-LC'] = model_df.loc[section_rows('ROSCOE-LC')]

    return summary_sections


def generate_single_dataset_final_summary_table(
    correls_by_section: Dict[str, Union[Dict, pandas.DataFrame]],
    baseline_set: str,
) -> pandas.DataFrame:

    emb_section_summaries = {}
    for emb_section in ('ROSCOE-SA', 'ROSCOE-SS'):
        section_data = []
        section_row_titles = []
        for model, model_df in correls_by_section[emb_section].items():
            if 'roscoe-512-roberta-base' in model:
                model_num = '(1)'
            elif 'all-mpnet-base-v2' in model:
                model_num = '(2)'
            elif 'sup-simcse-roberta-base' in model:
                model_num = '(3)'
            else:
                raise NotImplementedError()

            section_data.append(model_df['ANY_ERROR'][emb_section])
            section_row_titles.append(f"{emb_section} {model_num}")

        emb_section_summaries[emb_section] = pandas.DataFrame(
            data=section_data,
            index=section_row_titles,
            columns=['ANY_ERROR'],
        )

    summary_df = pandas.concat(
        [
            correls_by_section["baselines"].loc[REPORTED_BASELINES[baseline_set]],
            emb_section_summaries["ROSCOE-SA"],
            emb_section_summaries["ROSCOE-SS"],
            correls_by_section["ROSCOE-LI"].loc[['ROSCOE-LI']],
            correls_by_section["ROSCOE-LC"].loc[['ROSCOE-LC']],
        ]
    )
    return summary_df


def generate_single_dataset_final_summary_table_max(
    granular_sections: Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]],
    pvals_by_section: Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]],
):
    columns = list(STEP_COLUMN_RENAMES.keys())

    emb_section_summaries = {}
    for emb_section in ('ROSCOE-SA', 'ROSCOE-SS'):
        section_data = []
        section_row_titles = []
        for model, model_df in granular_sections[emb_section].items():
            if 'roscoe-512-roberta-base' in model:
                model_num = '(1)'
            elif 'all-mpnet-base-v2' in model:
                model_num = '(2)'
            elif 'sup-simcse-roberta-base' in model:
                model_num = '(3)'
            else:
                raise NotImplementedError()

            section_data.append(model_df[columns].max(axis=1)[emb_section])
            section_row_titles.append(f"{emb_section} {model_num}")

        emb_section_summaries[emb_section] = pandas.DataFrame(
            data=section_data,
            index=section_row_titles,
            columns=['MAX'],
        )

    def _max_df(df):
        return pandas.DataFrame({'MAX': df[columns].max(axis=1)})

    sig_baselines = granular_sections["baselines"].loc[REPORTED_BASELINES][
        pvals_by_section["baselines"] < 0.05
    ]

    summary_df = pandas.concat(
        [
            _max_df(sig_baselines),
            emb_section_summaries["ROSCOE-SA"],
            emb_section_summaries["ROSCOE-SS"],
            _max_df(granular_sections["ROSCOE-LI"].loc[['ROSCOE-LI']]),
            _max_df(granular_sections["ROSCOE-LC"].loc[['ROSCOE-LC']]),
        ]
    )
    return summary_df


CAN_REPEAT = {
    "faithfulness",
    "faithfulness_ww",
    "informativeness_step",
    "repetition_word",
    "stepwise_faithfulness",
    "stepwise_faithfulness_ww",
    "stepwise_informativeness_step",
    "stepwise_repetition_word",
    "step_0_faithfulness",
    "step_0_faithfulness_ww",
    "step_0_informativeness_step",
    "step_0_repetition_word",
    "middle_stepwise_faithfulness",
    "middle_stepwise_faithfulness_ww",
    "middle_stepwise_informativeness_step",
    "middle_stepwise_repetition_word",
    "last_faithfulness",
    "last_faithfulness_ww",
    "last_informativeness_step",
    "last_repetition_word",
    "informativeness_chain",
    "repetition_step",
    "stepwise_informativeness_chain",
    "stepwise_repetition_step",
    "step_0_informativeness_chain",
    "step_0_repetition_step",
    "middle_stepwise_informativeness_chain",
    "middle_stepwise_repetition_step",
    "last_informativeness_chain",
    "last_repetition_step",
}


def write_table_to_file(
    file_path: str, rows_by_section: Dict[str, List[List[str]]]
) -> None:
    with open(file_path, 'w') as f:
        rows_done = set()
        for title, table in rows_by_section.items():
            for l in table:
                row_title = l[0]
                if "ROSCOE-SS" in row_title or "ROSCOE-SA" in row_title:
                    f.write(f'{title}\n')
                elif "ROSCOE" in row_title:
                    f.write('\n')
                if row_title in rows_done and row_title not in CAN_REPEAT:
                    continue
                f.write('\t'.join(l) + '\n')
                rows_done.add(row_title)


def write_correls_and_sizes_tables(
    out_dir: str, base_file_name: str, correl_tables: Dict[str, List[List[str]]]
) -> None:
    out_correls = os.path.join(out_dir, f"{base_file_name}.CORRELS.txt")
    write_table_to_file(out_correls, correl_tables)
    print(f"Correlations written to: {out_correls}")


def process_values(
    human_and_scores_rows: List[List[str]], score_names: List[str]
) -> Dict[str, List[float]]:
    human_and_scores_column_titles = human_and_scores_rows[0]
    value_title_column_idx = {}
    value_titles = []

    def _question_from_column_title(t: str) -> str:
        if t.endswith('_result'):
            t = t.replace('_result', '_question')
        return human_and_scores_rows[1][human_and_scores_column_titles.index(t)]

    values_by_title = defaultdict(list)
    for i, t in enumerate(human_and_scores_column_titles):
        if (
            '_result' in t and '0_full_commonsenseSituationClaim' not in t
        ) or t in score_names:
            value_title_column_idx[t] = i
            value_titles.append(t)
            values_by_title[t] = []
    # make sure there are no duplicates
    value_titles = list(set(value_titles))

    human_column_titles, _ = split_list(value_titles, is_human_label_column_title)
    stepwise_labels = [h for h in human_column_titles if is_label_stepwise(h)]
    stepwise_label_roots = set(
        root_label_from_numbered_step_label(l) for l in stepwise_labels
    )

    label_step_numbers = step_numbers_from_step_labels(stepwise_labels)
    step_number_counts = {str(n): 0 for n in label_step_numbers}

    for row_idx, row in enumerate(human_and_scores_rows):
        if row_idx == 0:
            # skip column title row
            continue
        for title in value_titles:
            # for labels, 1.0 is good and 0.0 indicates a reasoning error
            value = row[value_title_column_idx[title]]
            if value == '':
                values_by_title[title].append(math.nan)
            elif "no" not in value and "yes" not in value:
                # this is either a likert label or metric score
                values_by_title[title].append(float(value))
            elif any(p in title for p in POSITIVE_ANNOTATIONS):
                if value == "no":
                    values_by_title[title].append(0.0)
                elif value == "yes":
                    values_by_title[title].append(1.0)
                else:
                    raise NotImplementedError(f'{title} value {value} not handled.')
            elif any(n in title for n in NEGATIVE_ANNOTATIONS):
                if value == "no":
                    values_by_title[title].append(1.0)
                elif value == "yes":
                    values_by_title[title].append(0.0)
                else:
                    raise NotImplementedError(f'{title} value {value} not handled.')
            else:
                raise NotImplementedError(
                    f'Unhandled label. Would answering the question "yes" be negative or positive?'
                    + '| {title}: {_question_from_column_title(title)}'
                )

            assert (
                len(values_by_title[title]) == row_idx
            ), f"no value was added for {title}"

        step_texts = {
            n: row[human_and_scores_column_titles.index(f'{n}_step_text')]
            for n in label_step_numbers
        }
        current_row_step_numbers = [n for n in label_step_numbers if step_texts[n]]
        last_step_number = current_row_step_numbers[-1]

        # update stats
        for n in current_row_step_numbers:
            step_number_counts[str(n)] += 1

        # this question only applies to the last step, so remove it from all other steps
        for i in range(1, last_step_number):
            label_title = f"{i}_step_step_questions_newFinalAnswerWrong_result"
            if label_title in values_by_title:
                values_by_title[label_title][-1] = math.nan

        # aggregates over steps
        for root in stepwise_label_roots:
            values_by_title[f'any_{root}'].append(
                min(
                    v
                    for v in [
                        values_by_title[f"{n}_{root}"][-1] for n in label_step_numbers
                    ]
                    if not math.isnan(v)
                )
            )

        # aggregate over steps and labels
        all_step_values = [
            values_by_title[f"{n}_{root}"][-1]
            for root in stepwise_label_roots
            for n in label_step_numbers
        ]
        values_by_title['any_step_any_issue'].append(
            min(v for v in all_step_values if not math.isnan(v))
        )

        binary_seq_lvl_questions = [
            f"0_{l}_result"
            for l in (
                "full_missingStep",
                "full_consistentSelf",
                "full_consistentContext",
                "full_newClearContradiction",
                "full_newMissingStep",
            )
        ] + ['any_step_any_issue']
        all_sequence_level_errors = [
            l
            for l in values_by_title.keys()
            if any(v in l for v in binary_seq_lvl_questions)
        ]
        values_by_title['ANY_ERROR'].append(
            min(values_by_title[e][-1] for e in all_sequence_level_errors)
        )

    assert_all_elements_same_length(
        values_by_title.values(),
        "some columns have a different number of rows",
    )

    print("Step number stats:")
    for k, v in step_number_counts.items():
        print(f"{k}:\t{v}")

    # remove columns with only NaNs
    values_by_title = remove_empty_columns(values_by_title)

    return values_by_title


def parse_all_scores(
    human_fn: str,
    baseline_fn: str,
    alignment_fn: str,
    removed_ids: List[str],
    reference_option: str,
    outfn: Optional[str] = None,
):
    """
    Uses metadata_example_idx to help identify examples that were marked problematic by
    reviewers.
    """
    rows, tags = parse_csv_with_idx(human_fn)

    if baseline_fn:
        newrows, tsvtags = parse_scorers_baseline(baseline_fn, rows)
    else:
        newrows = rows
        tsvtags = []
    newrows, tsvtags = parse_alignment_scorers(alignment_fn, newrows, tsvtags)

    all_tags = tsvtags
    all_tags = clean_tags(all_tags, reference_option)

    lines = []
    column_titles = tags + tsvtags
    lines.append(column_titles)
    for k, v in newrows.items():
        if v["metadata_example_idx"] in removed_ids:
            continue
        lines.append([newrows[k][t] for t in column_titles])

    print(f"ended up with {len(lines) - 1} rows.")

    if outfn is not None:
        with open(outfn, "w") as f:
            for l in lines:
                f.write('\t'.join(l) + '\n')
        print(f"Scores written to: {outfn}")
    return lines, all_tags


def read_and_process_data(
    human_annotation_filename: str,
    baseline_scores_filename: str,
    roscoe_scores_filename: str,
    removed_ids: List[str],
    reference_option: str,
    output_filename: Optional[str] = None,
) -> Tuple[Dict[str, List[float]], Set[str], Set[str]]:
    """
    Parses scores scores and annotations from their files, removes designated examples,
    and aggregates scores in various ways.

    Returns the processed scores, along with whatever new aggregate score/annotation
    names resulted from the processing.
    """
    rows_base, tags_base = parse_all_scores(
        human_fn=human_annotation_filename,
        baseline_fn=baseline_scores_filename,
        alignment_fn=roscoe_scores_filename,
        removed_ids=removed_ids,
        reference_option=reference_option,
        outfn=output_filename,
    )
    values_by_title = process_values(rows_base, tags_base)

    return values_by_title


def generate_scores_and_correls_across_datasets(
    correlation_type: str,
    out_dir: str,
    score_set: str,
    reference_option: str,
    adjust_ranges: bool,
):
    """
    Finds correlations between ROSCOE and baseline scores with annotations, where the
    data is taken from all datasets.

    Correlations are still split by sentence embedding model.
    """
    all_datasets = list(DATASET_TO_CONFIG.keys())

    all_configs_per_model = defaultdict(list)
    for dataset in all_datasets:
        config = DATASET_TO_CONFIG[dataset]
        for alignment_scores_filename in config['alignment_scores'][score_set]:
            model_name = alignment_scores_filename.split("/")[-2]
            all_configs_per_model[model_name].append(
                {
                    'labels': config['human_labels']['labels'],
                    'baseline_scores': config['baseline_scores'],
                    'alignment_scores': alignment_scores_filename,
                    'skip_indices': config['human_labels']['skip_indices'],
                }
            )

    correl_tables = {}
    pval_tables = {}
    sample_size_tables = {}
    for model, model_configs in all_configs_per_model.items():
        all_values_by_title = defaultdict(list)
        all_agg_stepwise_score_names = set()
        all_agg_stepwise_label_names = set()
        for config in model_configs:
            values_by_title = read_and_process_data(
                human_annotation_filename=config['labels'],
                baseline_scores_filename=config['baseline_scores'],
                roscoe_scores_filename=config['alignment_scores'],
                removed_ids=config['skip_indices'],
                reference_option=reference_option,
                adjust_ranges=adjust_ranges,
            )
            if len(all_values_by_title.keys()) == 0:
                for t in values_by_title.keys():
                    all_values_by_title[t] = values_by_title[t]
            else:
                if len(all_values_by_title.keys()) != len(values_by_title.keys()):
                    # remove labels that aren't in all datasets
                    diff = set(all_values_by_title.keys()) ^ set(values_by_title.keys())
                    for l in diff:
                        if l in all_values_by_title:
                            del all_values_by_title[l]
                for t in all_values_by_title.keys():
                    all_values_by_title[t].extend(values_by_title[t])

        somersd_table, pval_table, sizes_table = make_correlations_table(
            values_by_title=all_values_by_title,
            agg_stepwise_label_names=all_agg_stepwise_label_names,
            agg_stepwise_score_names=all_agg_stepwise_score_names,
            correlation_type=correlation_type,
        )

        # format for display
        somersd_table[0] = rename_columns_for_display(somersd_table[0])
        pval_table[0] = rename_columns_for_display(pval_table[0])
        sizes_table[0] = rename_columns_for_display(sizes_table[0])

        # write this model's correlation table to a file
        base_file_name = f"{model}_across_{'_'.join(all_datasets)}"
        write_correls_and_sizes_tables(
            out_dir=out_dir,
            base_file_name=base_file_name,
            correl_tables={model: somersd_table},
        )

        correl_tables[model] = somersd_table
        pval_tables[model] = pval_table
        sample_size_tables[model] = sizes_table

    correl_tables_df = {
        m: labeled_table_rows_to_dataframe(t) for m, t in correl_tables.items()
    }
    pval_tables_df = {
        m: labeled_table_rows_to_dataframe(t) for m, t in pval_tables.items()
    }
    granular_sections, _ = granular_summary(
        correls_by_model=correl_tables_df, pvals_by_model=pval_tables_df
    )
    write_granular_summary(
        dataset_name='_'.join(all_datasets),
        granular_sections=granular_sections,
        out_dir=out_dir,
    )

    # Write a massive correlation table file with all the model tables from above concatenated together.
    base_file_name = f"all_scores_across_{'_'.join(all_datasets)}"
    write_correls_and_sizes_tables(
        out_dir=out_dir, base_file_name=base_file_name, correl_tables=correl_tables
    )


def labeled_table_rows_to_dataframe(table_rows: List[List[str]]) -> pandas.DataFrame:
    return pandas.DataFrame(
        data=[r[1:] for r in table_rows[1:]],
        index=[r[0] for r in table_rows[1:]],
        columns=table_rows[0][1:],
    )


def generate_scores_and_correls_for_dataset(
    dataset_name: str,
    out_dir: str,
    reference_option: str,
    include_baselines: bool,
) -> Optional[Tuple[Dict, pandas.DataFrame]]:
    """
    Finds correlations between ROSCOE and baseline scores with annotations for one
    dataset.
    """
    config = DATASET_TO_CONFIG[dataset_name]
    baseline_scores_filename = None
    if include_baselines:
        baseline_scores_filename = config['baseline_scores']

    correl_tables = {}
    pval_tables = {}

    for alignment_scores_filename in config['alignment_scores']:
        model_name = alignment_scores_filename.split("/")[-2]
        file_name = dataset_name + "_all_scores_" + model_name
        out_scores = os.path.join(
            out_dir,
            file_name + ".txt",
        )
        values_by_title = read_and_process_data(
            human_annotation_filename=config['human_labels']['labels'],
            baseline_scores_filename=baseline_scores_filename,
            roscoe_scores_filename=alignment_scores_filename,
            removed_ids=config['human_labels']['skip_indices'],
            reference_option=reference_option,
            output_filename=out_scores,
        )
        somersd_table, pval_table = make_correlations_table(
            values_by_title=values_by_title
        )

        # format for display
        somersd_table[0] = rename_columns_for_display(somersd_table[0])
        pval_table[0] = rename_columns_for_display(pval_table[0])

        correl_tables[model_name] = somersd_table
        pval_tables[model_name] = pval_table

    base_file_name = f"{dataset_name}_all_scores"
    write_correls_and_sizes_tables(
        out_dir=out_dir, base_file_name=base_file_name, correl_tables=correl_tables
    )

    correl_tables_df = {
        m: labeled_table_rows_to_dataframe(t) for m, t in correl_tables.items()
    }
    pval_tables_df = {
        m: labeled_table_rows_to_dataframe(t) for m, t in pval_tables.items()
    }
    granular_sections, pvals_by_section = granular_summary(
        correls_by_model=correl_tables_df, pvals_by_model=pval_tables_df
    )
    write_granular_summary(
        dataset_name=dataset_name,
        granular_sections=granular_sections,
        out_dir=out_dir,
    )
    write_granular_summary_tex(
        dataset_name=dataset_name,
        granular_sections=granular_sections,
        pvals_by_section=pvals_by_section,
        out_dir=out_dir,
    )
    summary_sections = single_dataset_summary_max(
        correls_by_model=correl_tables_df, pvals_by_model=pval_tables_df
    )
    final_summary_df = generate_single_dataset_final_summary_table_max(
        granular_sections=granular_sections, pvals_by_section=pvals_by_section
    )
    return summary_sections, final_summary_df


def main(args):
    # make sure output directory exists
    pathlib.Path(os.path.dirname(args.out_dir)).mkdir(parents=True, exist_ok=True)

    dataset_list = (
        list(DATASET_TO_CONFIG.keys()) if args.dataset == 'all' else [args.dataset]
    )
    summaries_by_dataset = {}
    for dataset_name in dataset_list:
        print(f"Dataset: {dataset_name}")
        summaries_by_dataset[dataset_name] = generate_scores_and_correls_for_dataset(
            dataset_name=dataset_name,
            out_dir=args.out_dir,
            reference_option=args.reference_scores,
            include_baselines=args.include_baselines,
        )
    generate_every_dataset_summary_table(
        summaries_by_dataset={m: s for m, (s, _) in summaries_by_dataset.items()},
        summary_column='MAX',
        out_dir=args.out_dir,
    )
    if args.dataset == 'all':
        write_every_dataset_summary_table_tex(
            summaries_by_dataset={m: s for m, (s, _) in summaries_by_dataset.items()},
            out_dir=args.out_dir,
        )
    generate_every_dataset_final_summary_table(
        summaries_by_dataset={m: s for m, (_, s) in summaries_by_dataset.items()},
        summary_column='MAX',
        out_dir=args.out_dir,
    )
    if args.dataset == 'all':
        write_every_dataset_final_summary_table_tex(
            summaries_by_dataset={m: s for m, (_, s) in summaries_by_dataset.items()},
            out_dir=args.out_dir,
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        type=str,
        choices=list(DATASET_TO_CONFIG.keys()) + ['all'],
        default='all',
        help='name of the dataset',
    )
    parser.add_argument(
        '--out-dir', type=str, default=DEFAULT_OUTPUT_PATH, help='output directory'
    )
    parser.add_argument(
        '--reference-scores',
        type=str,
        choices=['no', 'yes', 'both'],
        default='both',
        help='use reference-based scores or reference-free',
    )
    parser.add_argument(
        '--include-baselines',
        type=bool,
        default=True,
        help='Whether to include baseline scores in correlations',
    )
    args = parser.parse_args()

    main(args)
