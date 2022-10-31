#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import csv
import math
import os
import pandas
from typing import Dict, List, Union

from projects.roscoe.score import (
    FAITHFUL_SENT,
    FAITHFUL_WORD,
    INFORM_STEP,
    REPETITION_WORD,
    INFORM_CHAIN,
    REPETITION_SENT,
    COHERENCE_STEP_VS_STEP,
    DISCOURSE_REPRESENTATION,
    GRAMMAR_STEP,
    PPL_STEP,
    PPL_CHAIN,
)

MODELS = ['roscoe-512-roberta-base', 'all-mpnet-base-v2', 'sup-simcse-roberta-base']

STEP_COLUMN_RENAMES = {
    "any_newGrammar": "GRAM",
    "any_newContradictContext": "FACT",
    "any_newExtraUselessInfo": "HALL",
    "any_newIntermediateFactualInfo": "RED",
    "any_newDroppableStep": "REP",
    "newMissingStep": "MISS",
    "any_newLogicalDeduction": "LOGIC",
    "any_newWorldKnowledge": "COMMON",
    "any_newMathError": "MATH",
}

COLUMN_RENAMES = {"newOverall": "QUAL", "newCoherent": "COH"}
COLUMN_RENAMES.update(STEP_COLUMN_RENAMES)

LATEX_ROW_MAPPING = {
    "rouge_1": "Rouge-1",
    "rouge_2": "Rouge-2",
    "rouge_l": "Rouge-L",
    "bleurt": "BLEURT",
    "bertScore_f": "BERTScore",
    "bartScore_f": "\\bartscore",
    "bartscore_cnn_para_f": "\\bartscoreParaCnn",
    "bartscore_finetuned_f": "\\bartscoreFineTuned",
    "prism_avg": "PRISM",
    "ctc_relevance_summary": "CTC-Relevance",
    "ctc_consistency_summary": "CTC-Consistency",
    "ROSCOE-SA": "\\ourmodelsa",
    "ROSCOE-SA (1)": "\\ourmodelsa",
    "ROSCOE-SA (2)": "\\ourmodelsa$^1",
    "ROSCOE-SA (3)": "\\ourmodelsa$^2",
    FAITHFUL_SENT: "\\ \\ \\ \\ \\ Faithfulness-Step",
    FAITHFUL_WORD: "\\ \\ \\ \\ \\ Faithfulness-Token",
    INFORM_STEP: "\\ \\ \\ \\ \\ Info-Step",
    REPETITION_WORD: "\\ \\ \\ \\ \\ Repetition-Token",
    "ROSCOE-SS": "\\ourmodelss",
    "ROSCOE-SS (1)": "\\ourmodelss",
    "ROSCOE-SS (2)": "\\ourmodelss$^1",
    "ROSCOE-SS (3)": "\\ourmodelss$^2",
    INFORM_CHAIN: "\\ \\ \\ \\ \\ Info-Chain",
    REPETITION_SENT: "\\ \\ \\ \\ \\ Repetition-Step",
    "ROSCOE-LI": "\\ourmodelli",
    COHERENCE_STEP_VS_STEP: "\\ \\ \\ \\ \\ Self-Consistency",
    DISCOURSE_REPRESENTATION: "\\ \\ \\ \\ \\ Source-Consistency",
    "ROSCOE-LC": "\\ourmodellc",
    GRAMMAR_STEP: "\\ \\ \\ \\ \\ Grammar",
    PPL_STEP: "\\ \\ \\ \\ \\ Perplexity-Step",
    PPL_CHAIN: "\\ \\ \\ \\ \\ Perplexity-Chain",
}

ROSCOE_SECTIONS = {
    'ROSCOE-SA': [
        'faithfulness',
        'faithfulness_ww',
        'informativeness_step',
        'repetition_word',
    ],
    'ROSCOE-SS': ['informativeness_chain', 'repetition_step'],
    'ROSCOE-LI': ['discourse_representation', 'coherence_step_vs_step'],
    'ROSCOE-LC': ['grammar_step', 'perplexity_step', 'perplexity_chain'],
}

DATASET_COLUMN_ORDER = ['drop', 'gsm8k', 'esnli', 'cosmos', 'semeval']

ANNOTATION_COLUMN_ORDER = [
    "QUAL",
    "COH",
    "COMMON",
    "FACT",
    "HALL",
    "RED",
    "REP",
    "LOGIC",
    "MATH",
    "GRAM",
    "MISS",
]

TO_CSV_TEX_KARGS = dict(
    float_format='{:.3f}'.format,
    sep='&',
    quotechar=' ',
    quoting=csv.QUOTE_ALL,
    line_terminator=' \\\\\n',
)


def section_rows(section: str) -> List[str]:
    return [section] + ROSCOE_SECTIONS[section]


def model_section_title_tex(section: str, model: str) -> str:
    if model == 'roscoe-512-roberta-base':
        model_name = "finetuned \\textit{sup-simcse-roberta-base}"
    elif model == 'all-mpnet-base-v2':
        model_name = "\\textit{all-mpnet-base-v2}"
    elif model == 'sup-simcse-roberta-base':
        model_name = "\\textit{sup-simcse-roberta-base}"
    else:
        raise NotImplementedError()
    return f"{LATEX_ROW_MAPPING[section]}~with {model_name} sentence embeddings"


def dataset_contains_math(dataset: str) -> bool:
    return any(dataset.lower() == d for d in ('drop', 'gsm8k', 'esnli'))


def top_2_format_replacements(df, absolute=False):
    to_format = {}
    for column in df.columns:
        to_format[column] = {}
        col_vals = df[column].tolist()
        if all(v == '-' or math.isnan(v) for v in col_vals):
            continue
        if absolute:
            col_vals = [abs(v) for v in col_vals]
        col_vals = set(col_vals)
        first, second = list(reversed(sorted(col_vals)[-2:]))
        to_format[column][first] = f"\\textbf{{{first:.3f}}}"
        to_format[column][second] = f"\\underline{{{second:.3f}}}"
        if absolute:
            neg_first = -1 * first
            neg_second = -1 * second
            to_format[column][neg_first] = f"\\textbf{{{neg_first:.3f}}}"
            to_format[column][neg_second] = f"\\underline{{{neg_second:.3f}}}"
    return to_format


def enforce_decimal_places(value: Union[str, float]) -> str:
    if not isinstance(value, str) or (
        isinstance(value, str) and '{' not in value and '-' not in value
    ):
        return f"{float(value):.3f}"
    return value


def write_granular_summary_tex(
    dataset_name: str,
    granular_sections: Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]],
    pvals_by_section: Dict[str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]],
    out_dir: str,
):
    granular_sections = copy.deepcopy(granular_sections)
    pvals_by_section = copy.deepcopy(pvals_by_section)

    def _format_section(df):
        columns = ANNOTATION_COLUMN_ORDER.copy()
        if not dataset_contains_math(dataset_name):
            columns.remove('MATH')
        return df.rename(index=LATEX_ROW_MAPPING, columns=COLUMN_RENAMES)[columns]

    # First, concat all data into one giant table to find the top 2 values in each column to bold/underline
    all_data = []
    for section in ('baselines', 'ROSCOE-SA', 'ROSCOE-SS', 'ROSCOE-LI', 'ROSCOE-LC'):
        if section == 'baselines':
            # Take care of formatting first, so column names will be the same when we look for these top values later
            cur_section_df = _format_section(granular_sections[section])
            granular_sections[section] = cur_section_df
            pvals_by_section[section] = _format_section(pvals_by_section[section])
            all_data.append(cur_section_df)
        elif section in {'ROSCOE-SA', 'ROSCOE-SS'}:
            for model in MODELS:
                cur_section_df = _format_section(
                    granular_sections[section][model].loc[ROSCOE_SECTIONS[section]]
                )
                granular_sections[section][model] = cur_section_df
                pvals_by_section[section][model] = _format_section(
                    pvals_by_section[section][model]
                )
                all_data.append(cur_section_df)
        else:
            cur_section_df = _format_section(
                granular_sections[section].loc[ROSCOE_SECTIONS[section]]
            )
            granular_sections[section] = cur_section_df
            pvals_by_section[section] = _format_section(pvals_by_section[section])
            all_data.append(cur_section_df)
    format_top_2 = top_2_format_replacements(pandas.concat(all_data))

    def _replace_top_2(df):
        df = df.replace(format_top_2)
        df = df.replace(math.nan, '-')
        df = df.applymap(enforce_decimal_places)
        return df

    out_path = os.path.join(out_dir, f"{dataset_name}_summary_granular.tex")
    with open(out_path, 'w') as f:
        # baselines
        formatted_section = _replace_top_2(granular_sections['baselines'])
        sig_mask = pvals_by_section['baselines'] < 0.05
        formatted_section[sig_mask] = formatted_section[sig_mask].applymap(
            lambda x: f"{x}$\\dagger$"
        )
        f.write(formatted_section.to_csv(**TO_CSV_TEX_KARGS))
        # separator between baselines and ROSCOE scores
        f.writelines(
            [
                "\t\\midrule\n",
                "\\multicolumn{10}{l}{\\textbf{\\ourmodel~Metrics (\\textbf{reference-free} on $(\\bm{s},\\bm{h})$)}} \\\\\n"
                "\t\\hline\n",
            ]
        )
        # ROSCOE scores, by section
        for section in ('ROSCOE-SA', 'ROSCOE-SS', 'ROSCOE-LI', 'ROSCOE-LC'):
            if section in {'ROSCOE-SA', 'ROSCOE-SS'}:
                for model in MODELS:
                    f.write(
                        "\\multicolumn{10}{l}{"
                        + model_section_title_tex(section, model)
                        + "} \\\\\n"
                    )
                    formatted_section = _replace_top_2(
                        granular_sections[section][model]
                    )
                    sig_mask = pvals_by_section[section][model] < 0.05
                    formatted_section[sig_mask] = formatted_section[sig_mask].applymap(
                        lambda x: f"{x}$\\dagger$"
                    )
                    f.write(formatted_section.to_csv(header=False, **TO_CSV_TEX_KARGS))
            else:
                f.write(
                    "\\multicolumn{10}{l}{" + LATEX_ROW_MAPPING[section] + "} \\\\\n"
                )
                formatted_section = _replace_top_2(granular_sections[section])
                sig_mask = pvals_by_section[section] < 0.05
                formatted_section[sig_mask] = formatted_section[sig_mask].applymap(
                    lambda x: f"{x}$\\dagger$"
                )
                f.write(formatted_section.to_csv(header=False, **TO_CSV_TEX_KARGS))
            f.write("\t\\midrule\n")

    print(f"Granular summary of {dataset_name} written to: {out_path}")


def write_every_dataset_summary_table_tex(
    summaries_by_dataset: Dict[
        str, Union[pandas.DataFrame, Dict[str, pandas.DataFrame]]
    ],
    out_dir: str,
):
    def _format_section(df):
        return df[DATASET_COLUMN_ORDER].rename(index=LATEX_ROW_MAPPING)

    table_sections = ('baselines', 'ROSCOE-SA', 'ROSCOE-SS', 'ROSCOE-LI', 'ROSCOE-LC')
    summaries_by_section = {}
    # First, concat all data into one giant table to find the top 2 values in each column to bold/underline
    all_data = []
    for section in table_sections:
        if section == 'baselines':
            cur_section_df = pandas.concat(
                [
                    sections[section].rename(columns={'MAX': name})
                    for name, sections in summaries_by_dataset.items()
                    if sections is not None
                ],
                axis=1,
            )
            # Take care of formatting first, so column names will be the same when we look for these top values later
            cur_section_df = _format_section(cur_section_df)
            summaries_by_section[section] = cur_section_df
            all_data.append(cur_section_df)
        elif section in {'ROSCOE-SA', 'ROSCOE-SS'}:
            summaries_by_section[section] = {}
            for model in MODELS:
                cur_section_df = pandas.concat(
                    [
                        sections[section][model]
                        .loc[ROSCOE_SECTIONS[section]]
                        .rename(columns={'MAX': name})
                        for name, sections in summaries_by_dataset.items()
                        if sections is not None
                    ],
                    axis=1,
                )
                cur_section_df = _format_section(cur_section_df)
                summaries_by_section[section][model] = cur_section_df
                all_data.append(cur_section_df)
        else:
            cur_section_df = pandas.concat(
                [
                    sections[section]
                    .loc[ROSCOE_SECTIONS[section]]
                    .rename(columns={'MAX': name})
                    for name, sections in summaries_by_dataset.items()
                    if sections is not None
                ],
                axis=1,
            )
            cur_section_df = _format_section(cur_section_df)
            summaries_by_section[section] = cur_section_df
            all_data.append(cur_section_df)
    format_top_2 = top_2_format_replacements(pandas.concat(all_data))

    def _replace_top_2(df):
        df = df.replace(format_top_2)
        df = df.replace(math.nan, '-')
        df = df.applymap(enforce_decimal_places)
        return df

    out_path = os.path.join(out_dir, "summary_granular.tex")
    with open(out_path, 'w') as f:
        # baselines section
        f.write(
            _replace_top_2(summaries_by_section['baselines']).to_csv(**TO_CSV_TEX_KARGS)
        )

        # separator between baselines and ROSCOE scores
        f.writelines(
            [
                "\\midrule\n",
                "\\multicolumn{6}{l}{\\textbf{\\ourmodel~Metrics (\\textbf{reference-free} on $(\\bm{s},\\bm{h})$)}} \\\\\n"
                "\\hline\n",
            ]
        )

        for section in table_sections[1:]:
            if section in {'ROSCOE-SA', 'ROSCOE-SS'}:
                for model in MODELS:
                    f.write(
                        "\\multicolumn{6}{l}{"
                        + model_section_title_tex(section, model)
                        + "} \\\\\n"
                    )
                    f.write(
                        _replace_top_2(summaries_by_section[section][model]).to_csv(
                            header=False, **TO_CSV_TEX_KARGS
                        )
                    )
            else:
                f.write(
                    "\\multicolumn{6}{l}{" + LATEX_ROW_MAPPING[section] + "} \\\\\n"
                )
                f.write(
                    _replace_top_2(summaries_by_section[section]).to_csv(
                        header=False, **TO_CSV_TEX_KARGS
                    )
                )
            f.write("\\midrule\n")

    print(f"Summary written to: {out_path}")


def write_every_dataset_final_summary_table_tex(
    summaries_by_dataset: Dict[str, pandas.DataFrame],
    out_dir: str,
):
    summary_all_datasets = pandas.concat(
        [
            df.rename(columns={'MAX': name})
            for name, df in summaries_by_dataset.items()
            if df is not None
        ],
        axis=1,
    )
    formatted_summary = summary_all_datasets[DATASET_COLUMN_ORDER].rename(
        index=LATEX_ROW_MAPPING
    )

    to_format = top_2_format_replacements(formatted_summary)
    formatted_summary = formatted_summary.replace(to_format)
    formatted_summary = formatted_summary.replace(math.nan, '-')
    formatted_summary = formatted_summary.applymap(enforce_decimal_places)

    summary_out_path = os.path.join(out_dir, "summary.tex")
    formatted_summary.to_csv(summary_out_path, **TO_CSV_TEX_KARGS)
    print(f"Summary written to: {summary_out_path}")
