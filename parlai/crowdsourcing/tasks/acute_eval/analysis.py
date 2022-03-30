#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
ACUTE-Eval Analyzer.

FOR ANALYSIS!!
"""

import hashlib
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from IPython.core.display import HTML
from mephisto.tools.data_browser import DataBrowser as MephistoDataBrowser
from mephisto.abstractions.databases.local_database import LocalMephistoDB
from mephisto.data_model.unit import Unit as MephistoUnit
from mephisto.data_model.worker import Worker
from scipy.stats import binom_test

from parlai.core.params import ParlaiParser
from parlai.crowdsourcing.tasks.acute_eval.acute_eval_blueprint import (
    BLUEPRINT_TYPE as ACUTE_EVAL_BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.tasks.acute_eval.fast_acute_blueprint import (
    FAST_ACUTE_BLUEPRINT_TYPE,
)

# To register the ACUTE-Eval and Fast ACUTE blueprints
from parlai.crowdsourcing.tasks.acute_eval.util import get_hashed_combo_path

_ = ACUTE_EVAL_BLUEPRINT_TYPE
_ = FAST_ACUTE_BLUEPRINT_TYPE
# TODO: blueprint type strings need to be imported here to register the blueprints -
# find a better way to scale up when there are many more subclassed ACUTE blueprints

# throw away turkers below this threshold
AGREEMENT_THRESHOLD = 0.8
# do we count ties as agreements?
AGREEMENT_TIES_OKAY = False
# NOTE: these could be added as flags if desired


def setup_args():
    """
    Setup appropriate args.
    """
    parser = ParlaiParser(False, False)
    parser.add_argument(
        '-ids',
        '--run-ids',
        type=str,
        default=None,
        help='Comma-separated list of run IDs to analyze',
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default=None,
        help='Optional root ACUTE-Eval save directory',
    )
    parser.add_argument(
        '--outdir', type=str, default=None, help='Where to save the results'
    )
    parser.add_argument(
        '--pairings-filepath',
        type=str,
        default=None,
        help='Path to the ACUTE analysis pairs for the corresponding run id',
    )
    parser.add_argument(
        '--mephisto-root',
        type=str,
        default=None,
        help='Where to check for mephisto data (default own dir)',
    )
    parser.add_argument(
        '--model-ordering',
        type=str,
        default=None,
        help='Comma-separated list of models, in the order in which to display them',
    )
    return parser


class AcuteAnalyzer(object):
    """
    Analyzer.

    Given a run_id, we can do lots of fun things!
    """

    CHECKBOX_PREFIX = 'checkbox: '
    # Prepended to checkbox columns in self.dataframe

    def __init__(self, opt: Dict, remove_failed: bool = True):
        """
        Initialize the analyzer.

        Builds up the dataframe

        :param opt:
            opt dict

        :param remove_failed:
            Whether to remove ratings from turkers who failed onboarding
        """
        assert ',' not in opt['run_ids'], "AcuteAnalyzer can only handle one run ID!"
        self.run_id = opt['run_ids']
        self.pairings_filepath = opt['pairings_filepath']
        self.outdir = opt['outdir']
        self.root_dir = opt['root_dir']
        # Get task for loading pairing files
        self.task = opt.get('task', 'q')
        if opt.get('model_ordering') is not None:
            self.custom_model_ordering = opt['model_ordering'].split(',')
        else:
            self.custom_model_ordering = None
        if not self.outdir or not self.pairings_filepath:
            # Default to using self.root_dir as the root directory for outputs
            assert self.root_dir is not None and os.path.isdir(
                self.root_dir
            ), '--root-dir must be a real directory!'
        if not self.pairings_filepath:
            # Will be set to a non-empty path later
            self.pairings_filepath = ''
        if not self.outdir:
            self.outdir = os.path.join(self.root_dir, f'{self.run_id}-results')
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir, exist_ok=True)
        mephisto_root_path = opt['mephisto_root']
        if not mephisto_root_path:
            mephisto_root_path = None
        self.mephisto_db = LocalMephistoDB(database_path=mephisto_root_path)
        self.mephisto_data_browser = MephistoDataBrowser(db=self.mephisto_db)
        self.checkbox_prefix = self.CHECKBOX_PREFIX
        # Prepended to checkbox columns in self.dataframe
        self.dataframe = self._extract_to_dataframe()
        self._check_eval_question()
        if remove_failed:
            self._remove_failed_onboarding()
        if self.dataframe.index.size == 0:
            raise ValueError('No valid results found!')
        self._get_model_nick_names()
        self._load_pairing_files()

    def _extract_response_by_index(
        self, unit_details: Dict[str, Any], idx: int
    ) -> Optional[Dict[str, Any]]:
        """
        Extract response data from task data.

        :param unit_details:
            full extracted data from a unit
        :param idx:
            index of the singular evaluation within unit_details to extract

        :return response:
            Formatted worker's response data from the task
        """
        task_data = unit_details['data'][idx]
        response: Dict[str, Any] = {
            'run_id': self.run_id,
            'worker': unit_details['worker_id'],
            'worker_name': Worker.get(
                self.mephisto_db, unit_details['worker_id']
            ).worker_name,
            'time_taken': unit_details['task_end'] - unit_details['task_start'],
            'question': task_data['task_specs']['question'],
            'unit_id': unit_details['unit_id'],
            'task_start': unit_details['task_start'],
        }
        onboarding = task_data['task_specs'].get('is_onboarding', False)
        if 'speakerChoice' not in task_data or task_data['speakerChoice'] == '':
            print('speakerChoice not in task data!')
            return
        choice = task_data['speakerChoice']
        if onboarding:
            response['correct'] = choice == task_data['pairing_dict']['correct_answer']
        else:
            response['correct'] = -1

        speakers_to_eval = sorted(task_data["pairing_dict"]["speakers_to_eval"])
        response.update(
            {
                'winner': choice,
                'loser': speakers_to_eval[1 - (speakers_to_eval.index(choice))],
                'eval_choice_0': speakers_to_eval[0],
                'eval_choice_1': speakers_to_eval[1],
                'reason': task_data['textReason'],
                'is_onboarding': onboarding,
                'matchup': f"{'__vs__'.join(speakers_to_eval)}",
                'pairing_id': task_data['pair_id'],
            }
        )

        # If it exists, add in which checkboxes of possible reasons the Turkers checked
        if len(task_data.get('speakerReasons', {})) > 0:
            response.update(
                {
                    self.checkbox_prefix + reason: checked
                    for reason, checked in task_data['speakerReasons'].items()
                }
            )
        return response

    def _parse_unit(self, unit: MephistoUnit) -> Optional[Dict[str, Any]]:
        """
        Return data for a given unit.

        If the data is corrupt for whatever reason, we return None

        :param unit:
            MephistoUnit of what should be a completed task by a worker

        :return data:
            Optional dict with the task's formatted data
        """
        try:
            return self.mephisto_data_browser.get_data_from_unit(unit)
        except AssertionError:
            print(
                f"WARNING: Data for run_id `{self.run_id}` not found for "
                f"unit id {unit.db_id}"
            )
            return None

    def _extract_to_dataframe(self) -> pd.DataFrame:
        """
        Extract the data from the run to a pandas dataframe.
        """
        units = self.mephisto_data_browser.get_units_for_task_name(self.run_id)
        responses: List[Dict[str, Any]] = []
        for unit in units:
            unit_details = self._parse_unit(unit)
            if unit_details is None:
                continue
            for idx in range(len(unit_details['data'])):
                response = self._extract_response_by_index(unit_details, idx)
                if response is not None:
                    responses.append(response)

        if len(responses) == 0:
            raise ValueError('No valid results found!')
        else:
            return pd.DataFrame(responses)

    def _check_eval_question(self):
        """
        Check that the same eval question has been used for all results.
        """
        if len(set(self.dataframe['question'].unique())) > 1:
            raise ValueError(
                'All results must share the same eval question for consistency!'
            )

    def _remove_failed_onboarding(self):
        """
        Remove workers who failed onboarding.
        """
        df = self.dataframe

        all_workers_failing_onboarding = df.loc[
            df['is_onboarding'] & (df['correct'] == False), 'worker'  # noqa: E712
        ].values

        workers_failing_onboarding = sorted(
            np.unique(all_workers_failing_onboarding).tolist()
        )

        self.dataframe = df[
            ~df["worker"].isin(workers_failing_onboarding) & ~df["is_onboarding"]
        ]
        print(
            f'{self.dataframe.index.size:d} dataframe entries remaining after removing users who failed onboarding.'
        )

    def _load_pairing_files(self):
        df = self.dataframe
        if not os.path.exists(self.pairings_filepath):
            print('No valid pairings filepath was passed in: will extract likely path.')
            self.pairings_filepath = get_hashed_combo_path(
                root_dir=self.root_dir,
                subdir='pairings_files',
                task=self.task,
                combos=self.combos,
            )
        if not os.path.exists(self.pairings_filepath):
            print(
                f'WARNING: Pairings filepath {self.pairings_filepath} could not be found.'
            )
            self.pairings_filepath = os.path.join(
                self.root_dir,
                'pairings_files',
                hashlib.sha1(
                    '___vs___'.join(
                        [f"{m}.{'q'.replace(':', '_')}" for m in self.models]
                    ).encode('utf-8')
                ).hexdigest()[:10],
            )
        if not os.path.exists(self.pairings_filepath):
            # For backward compatibility
            print(
                f'WARNING: Pairings filepath {self.pairings_filepath} could not be found.'
            )
            self.pairings_filepath = os.path.join(
                self.root_dir,
                'pairings_files',
                '___vs___'.join(
                    [f"{m}.{self.task.replace(':', '_')}" for m in self.models]
                ),
            )
        if not os.path.exists(self.pairings_filepath):
            print(
                f'NOTE: Pairings filepath {self.pairings_filepath} could not be found!'
            )
            return
        self.pairings = []
        with open(self.pairings_filepath, 'r') as f:
            for line in f:
                pair = json.loads(line)
                model1, model2 = pair['speakers_to_eval']
                pair[model1] = pair['dialogue_dicts'][0]
                pair[model2] = pair['dialogue_dicts'][1]
                del pair['dialogue_dicts']
                self.pairings.append(pair)
        self.pairs_to_eval = [self.pairings[i] for i in df.pairing_id.values.tolist()]
        # Build dialogue_ids => dialogue mappings

        winner_dialogues = []
        loser_dialogues = []
        for i, (_, row) in enumerate(df.iterrows()):
            winner = row['winner']
            loser = row['loser']
            winner_dialogues.append(self.pairs_to_eval[i][winner])
            loser_dialogues.append(self.pairs_to_eval[i][loser])
        df['pairs_to_eval'] = pd.Series(self.pairs_to_eval, index=df.index)
        df['winner_dialogue'] = pd.Series(winner_dialogues, index=df.index)
        df['loser_dialogue'] = pd.Series(loser_dialogues, index=df.index)
        self.dataframe = df

    def _get_model_nick_names(self):
        df = self.dataframe
        df = df[df['run_id'] == self.run_id]
        matchups = list(df.matchup.unique())
        models = set()
        combos = set()
        for matchup in matchups:
            model1, model2 = matchup.split('__vs__')
            models.add(model1)
            models.add(model2)
            combos.add(tuple(sorted((model1, model2))))
        self.models = list(models)
        self.models.sort()
        self.combos = list(combos)
        self.combos.sort()

    def get_reasons(self) -> List[str]:
        """
        Return dataframe reasons.
        """
        return self.dataframe['reason'].values.tolist()

    def get_max_hits_per_worker(self) -> List[int]:
        """
        Get max number of hits per worker.
        """
        return self.dataframe.groupby('worker')['run_id'].count().max()

    def get_wins_per_model_matchup(self) -> pd.DataFrame:
        """
        Return the wins for each model by matchup.
        """
        self.matchup_total_df = (
            self.dataframe.groupby(['eval_choice_0', 'eval_choice_1'])['run_id']
            .count()
            .to_frame('matchup_total')
        )
        self.win_total_df = (
            self.dataframe.groupby(
                ['eval_choice_0', 'eval_choice_1', 'winner', 'loser']
            )['loser']
            .count()
            .to_frame('win_total')
            .reset_index()
            .set_index(['eval_choice_0', 'eval_choice_1'])
        )
        return self.win_total_df

    def get_win_fractions(self) -> pd.DataFrame:
        """
        Return the joined matchup + win totals, get win fractions.

        Sorted according to win percentage
        """
        if not hasattr(self, 'win_total_df'):
            self.get_wins_per_model_matchup()

        self.win_fraction_df = self.matchup_total_df.join(self.win_total_df).assign(
            win_frac=lambda df: df['win_total'] / df['matchup_total']
        )

        pivoted_df = self.win_fraction_df.pivot(
            index="loser", columns="winner", values="win_frac"
        )
        if self.custom_model_ordering is not None:
            # Use the ordering of the models supplied by the user
            assert set(self.custom_model_ordering) == set(pivoted_df.columns)
            self.model_ordering = self.custom_model_ordering
        else:
            self.model_ordering = (
                self.win_fraction_df.groupby("winner")["win_frac"]
                .mean()
                .sort_values()
                .index.values.tolist()
            )
        self.sorted_win_frac_df = pivoted_df.reindex(
            index=self.model_ordering, columns=self.model_ordering
        )
        return self.sorted_win_frac_df

    def get_num_hits_per_matchup(self):
        """
        Return the number of hits per matchup.
        """
        matchup_total_1_df = self.matchup_total_df.reset_index()
        matchup_total_2_df = matchup_total_1_df.rename(
            columns={'eval_choice_0': 'eval_choice_1', 'eval_choice_1': 'eval_choice_0'}
        )
        self.num_hits_per_matchup_df = (
            pd.concat([matchup_total_1_df, matchup_total_2_df], axis=0)
            .pivot(
                index='eval_choice_0', columns='eval_choice_1', values='matchup_total'
            )
            .reindex(index=self.model_ordering, columns=self.model_ordering)
        )
        return self.num_hits_per_matchup_df

    def _compile_checkbox_stats(self) -> Dict[str, pd.DataFrame]:
        """
        Return the fraction of time that Turkers selected each checkbox.

        Results are cut both (1) by matchup and winner and (2) by just the winner. Each
        checkbox represents one reason that the Turkers could have chosen the speaker
        that they did.
        """
        checkbox_columns = [
            col
            for col in self.dataframe.columns
            if col.startswith(self.checkbox_prefix)
        ]
        group_column_types = {
            'matchup_and_winner': ['matchup', 'winner'],
            'winner': ['winner'],
        }
        grouped_dataframes = {}
        for group_type, group_columns in group_column_types.items():
            selected_columns = (
                self.dataframe[group_columns + checkbox_columns]
                .rename(
                    columns={
                        col: col[len(self.checkbox_prefix) :]
                        for col in checkbox_columns
                    }
                )
                .set_index(group_columns)
                .fillna(False)
            )
            grouped_dataframes[group_type] = selected_columns.groupby(
                group_columns
            ).mean()
        return grouped_dataframes

    def _compile_convos_and_reasons(self) -> str:
        """
        Create a human-readable string of all pairs of conversations, as well as which
        conversation each Turker chose and their reason for choosing it.
        """

        pairing_outputs = []

        for _, pairing_sr in self.dataframe.iterrows():
            winning_dialogue = self._dialogue_to_string(
                pairing_sr['winner_dialogue']['dialogue']
            )
            loser_dialogue = self._dialogue_to_string(
                pairing_sr['loser_dialogue']['dialogue']
            )
            pairing_output = f"""CONVO PAIR ID: {pairing_sr['pairing_id']}

WINNING DIALOGUE: {pairing_sr['winner']}
{winning_dialogue}

LOSING DIALOGUE: {pairing_sr['loser']}
{loser_dialogue}

QUESTION: {pairing_sr['question']}
TURKER'S CHOICE: {pairing_sr['winner']}
REASON: {pairing_sr['reason']}



"""
            pairing_outputs.append(pairing_output)

        return ''.join(pairing_outputs)

    @staticmethod
    def _dialogue_to_string(dialogue: List[dict]) -> str:
        """
        Convert a list of dictionaries into a human-readable conversation.

        Each dictionary represents one utterance.
        """
        utterance_strings = []
        for utterance_dict in dialogue:
            if utterance_dict["id"] == "human_evaluator":
                speaker_string = "HUMAN"
            else:
                speaker_string = utterance_dict["id"]
            utterance = utterance_dict["text"]
            utterance_strings.append(f"[{speaker_string}]: {utterance}")
        return "\n".join(utterance_strings)

    def get_matchup_totals_with_significance(self) -> pd.DataFrame:
        """
        Return dataframe with matchup win totals + significance.
        """

        def _signf_level(p):
            if p < 0.001:
                return "***", "p<.001"
            elif p < 0.01:
                return "**", "p<.01"
            elif p < 0.05:
                return "*", "p<.05"
            else:
                return "", "p>.05"

        output = []
        for _, run_annotations in self.dataframe.groupby('run_id'):
            question = list(run_annotations.question)[0]
            for matchup, annotations in run_annotations.groupby('matchup'):
                model1, model2 = matchup.split('__vs__')
                wincount1 = np.sum(annotations['winner'] == model1)
                wincount2 = np.sum(annotations['winner'] == model2)
                numratings = wincount1 + wincount2
                winrate1 = np.mean(annotations['winner'] == model1)
                winrate2 = np.mean(annotations['winner'] == model2)
                p = binom_test([wincount1, wincount2])

                stars, plevel = _signf_level(p)

                agreements = []
                for _, pairing_annotations in annotations.groupby('pairing_id'):
                    pair_wincount1 = np.sum(pairing_annotations['winner'] == model1)
                    pair_wincount2 = np.sum(pairing_annotations['winner'] == model2)
                    if pair_wincount1 < 2 and pair_wincount2 < 2:
                        if pair_wincount1 == 1 and pair_wincount2 == 1:
                            agreements.append(0)
                    else:
                        majority_wincount = max(pair_wincount1, pair_wincount2)
                        num_pair_annotations = pair_wincount1 + pair_wincount2
                        pair_agreement = majority_wincount / num_pair_annotations
                        agreements.append(pair_agreement)
                total_agreement = np.mean(agreements)

                output.append(
                    {
                        'question': question,
                        'matchup': matchup,
                        'model1': model1,
                        'model2': model2,
                        'numwins1': wincount1,
                        'numwins2': wincount2,
                        'winrate1': winrate1,
                        'winrate2': winrate2,
                        'numratings': numratings,
                        'p': p,
                        'stars': stars,
                        'sigf': plevel,
                        'agree': total_agreement,
                    }
                )
        output = pd.DataFrame(output)
        # order the columns how we want
        self.significance_df = output[
            [
                'question',
                'matchup',
                'model1',
                'numwins1',
                'winrate1',
                'model2',
                'numwins2',
                'winrate2',
                'numratings',
                'sigf',
                'stars',
                'p',
                'agree',
            ]
        ]
        return self.significance_df

    def save_results(self, path: str = None):
        """
        Save results to a certain path.
        """
        if not hasattr(self, 'significance_df'):
            self.get_matchup_totals_with_significance()
        if path is None:
            path = self.outdir

        # Save raw dataframe
        self.dataframe.to_csv(f'{path}/{self.run_id}.full.csv', index=False)

        with open('{}/{}.significance.csv'.format(path, self.run_id), 'w') as f:
            f.write(self.significance_df.to_csv(index=False))
        print(
            'To visualize significance result, try cat {} | column -t -s, | less -S'.format(
                '{}/{}.significance.csv'.format(path, self.run_id)
            )
        )
        with open('{}/{}.grid.csv'.format(path, self.run_id), 'w') as f:
            f.write(self.get_win_fractions().to_csv(index=True))
        with open(f'{path}/{self.run_id}.grid.winners_as_rows.csv', 'w') as f:
            f.write(self.get_win_fractions().transpose().to_csv(index=True))
        print(
            'To visualize grid result, try cat {} | column -t -s, | less -S'.format(
                '{}/{}.grid.csv'.format(path, self.run_id)
            )
        )

        # Save stats on how many ratings each worker did
        ratings_per_worker = (
            self.dataframe.groupby('worker')['run_id']
            .count()
            .sort_values(ascending=False)
        )
        ratings_per_worker.to_csv(f'{path}/{self.run_id}.ratings_per_worker.csv')

        # Save stats on how often Turkers selected each checkbox that represents one
        # reason to pick the speaker they did
        if any(col.startswith(self.checkbox_prefix) for col in self.dataframe.columns):
            checkbox_stats_dataframes = self._compile_checkbox_stats()
            for group_type, stats in checkbox_stats_dataframes.items():
                stats.to_csv(f'{path}/{self.run_id}.checkbox_stats.{group_type}.csv')

        if not hasattr(self, 'pairings'):

            print('No pairing file found, skipping conversation visualizations.')

        else:

            with open('{}/{}.reason.html'.format(path, self.run_id), 'w') as f:
                f.write(render_conversations_per_matchups(self.dataframe, True).data)
            print(
                'To visualize conversations with reasons only result, '
                'try scp username@devfair:{} to your local machine'.format(
                    ' {}/{}.reason.html'.format(path, self.run_id)
                )
            )
            with open('{}/{}.all.html'.format(path, self.run_id), 'w') as f:
                f.write(render_conversations_per_matchups(self.dataframe, False).data)
            print(
                'To visualize conversations result, try scp username@devfair:{}'
                ' to your local machine'.format(
                    '{}/{}.all.html'.format(path, self.run_id)
                )
            )

            # Write all pairs of dialogues, as well as the Turkers' choices and reasons, as
            # a text file
            compiled_text = self._compile_convos_and_reasons()
            with open(f'{path}/{self.run_id}.all_convo_pairs.txt', 'w') as f:
                f.write(compiled_text)


class MultiRunAcuteAnalyzer(AcuteAnalyzer):
    """
    Combine results from different ACUTE-Eval runs.
    """

    def __init__(self, opt: Dict, dataframes: Dict[str, pd.DataFrame]):
        """
        Read in and combine the dataframes of other already-analyzed ACUTE-Eval runs.
        """

        self.outdir = opt['outdir']
        if opt.get('model_ordering') is not None:
            self.custom_model_ordering = opt['model_ordering'].split(',')
        else:
            self.custom_model_ordering = None
        self.run_id = 'combined'
        self.checkbox_prefix = self.CHECKBOX_PREFIX
        # Prepended to checkbox columns in self.dataframe

        for dataframe in dataframes.values():
            dataframe.loc[:, 'run_id'] = self.run_id
            # Overwrite the run_id so that results will combine across runs
        self.dataframe = pd.concat(dataframes.values(), axis=0)

        # Check that all results across all runs share the same eval question
        self._check_eval_question()


def get_multi_run_analyzer(opt) -> MultiRunAcuteAnalyzer:
    """
    Return an object to analyze the results of multiple runs simultaneously.

    Load HITs from each run into a separate dataframe, and then pass all dataframes into
    a separate analyzer class that will concatenate them.
    """

    run_ids = opt['run_ids'].split(',')

    # Define paths
    assert (
        opt['outdir'] is not None
    ), '--outdir must be specified when combining results of multiple runs!'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    opt['outdir'] = os.path.join(opt['outdir'], f'combined_runs_{timestamp}')
    os.makedirs(opt['outdir'], exist_ok=True)
    run_id_list_path = os.path.join(opt['outdir'], 'run_ids.txt')

    # Save a simple list of all run IDs stitched together
    with open(run_id_list_path, 'w') as f:
        for run_id in run_ids:
            f.write(run_id + '\n')

    # Loop loading HITs over all run ids into dataframes
    dataframes = {}
    for run_id in run_ids:
        print(f'\nStarting to load HITs for run ID {run_id}.')
        opt_copy = deepcopy(opt)
        opt_copy['run_ids'] = run_id
        dataframes[run_id] = AcuteAnalyzer(opt_copy).dataframe

    return MultiRunAcuteAnalyzer(opt=opt, dataframes=dataframes)


def render_row(row):
    result = []
    for i, turn in enumerate(row['winner_dialogue']['dialogue']):
        speakername = turn['id']
        text = turn['text']
        is_bot = (speakername != 'human_evaluator') and (speakername != 'other_speaker')
        if i > 2 and is_bot:
            speakername = 'bot'
        align = 'right' if is_bot else 'left'
        color = "white" if is_bot else "black"
        bgcolor = '#2391f7' if is_bot else '#e1e1e7'

        result.append(
            (
                '<div style="overflow: auto; padding: 1ex 0;">'
                '<div style="clear: both; float: {}; color: {}; background-color: {}; padding: 0.5em 1em; border-radius: 1em; max-width: 80%">'
                '<p style="margin: 0">{}: {}</p>'
                '</div>'
                '</div>'
            ).format(align, color, bgcolor, speakername, text)
        )
    winner_dialogue = (
        '<div style="background-color: white; margin: 0em; padding: 0.5em; '
        'font-family: sans-serif; font-size: 9pt; width: 99%;">'
        + ''.join(result)
        + '</div>'
    )
    result = []
    for i, turn in enumerate(row['loser_dialogue']['dialogue']):
        speakername = turn['id']
        is_bot = (speakername != 'human_evaluator') and (speakername != 'other_speaker')
        if i > 2 and is_bot:
            speakername = 'bot'
        text = turn['text']

        align = 'right' if is_bot else 'left'
        color = "white" if is_bot else "black"
        bgcolor = '#2391f7' if is_bot else '#e1e1e7'

        result.append(
            (
                '<div style="overflow: auto; padding: 1ex 0;">'
                '<div style="clear: both; float: {}; color: {}; background-color: {}; padding: 0.5em 1em; border-radius: 1em; max-width: 80%">'
                '<p style="margin: 0">{}: {}</p>'
                '</div>'
                '</div>'
            ).format(align, color, bgcolor, speakername, text)
        )
    loser_dialogue = (
        '<div style="background-color: white; margin: 0em; padding: 0.5em; '
        'font-family: sans-serif; font-size: 9pt; width: 99%;">'
        + ''.join(result)
        + '</div>'
    )

    return HTML(
        '<tr><td>{}</td><td>{}</td><td>{}</td></tr>'.format(
            winner_dialogue, loser_dialogue, row['reason']
        )
    )


def render_many_conversations(table):
    return HTML(
        '<table><tr><th>Winner Conversation</th><th>Loser Conversation</th><th>Reason</th></tr>{}</table>'.format(
            ''.join(render_row(row).data for i, row in table.iterrows())
        )
    )


def render_conversations_per_matchups(table, force_reasons=True):
    matchups = list(table.matchup.unique())
    result = ''
    if force_reasons:
        table = table[table['reason'] != '']
    for matchup in matchups:
        length = min(10, len(table[table['matchup'] == matchup]))
        result += '<h2>{}</h2><body>{}</body>'.format(
            matchup,
            render_many_conversations(table[table['matchup'] == matchup][:length]).data,
        )
    return HTML(result)


if __name__ == "__main__":

    parser = setup_args()
    opt_ = parser.parse_args()

    if ',' not in opt_['run_ids']:
        analyzer = AcuteAnalyzer(opt_)
    else:
        analyzer = get_multi_run_analyzer(opt_)
    analyzer.save_results()

    # Print win fractions
    results = pd.DataFrame(analyzer.get_win_fractions())
    print(results.round(2).to_string())

    # Print matchup totals with significance
    result_ = pd.DataFrame(analyzer.get_matchup_totals_with_significance())
    result_ = result_.drop(columns=['matchup', 'agree'])
    print(result_.round(2).to_string())
