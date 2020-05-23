#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
ACUTE-Eval Analyzer.

Given a run_id, one can print out and save the results.

If you specify the pairings filepath, you can visualize
the choices that turkers made about specific conversations.
"""
import os
import pandas as pd
import numpy as np

from typing import Dict, Any, List, Optional
from scipy.stats import binom_test

from parlai.mturk.core.mturk_data_handler import MTurkDataHandler
from parlai import __file__ as parlai_filepath
from parlai.core.params import ParlaiParser
import json
from IPython.core.display import HTML


# throw away turkers below this threshold
AGREEMENT_THRESHOLD = 0.8
# do we count ties as agreements?
AGREEMENT_TIES_OKAY = False


def setup_args():
    """
    Setup appropriate args.
    """
    parser = ParlaiParser(True, False)
    parser.add_argument(
        '-id', '--run-id', type=str, default=None, help='run id to analyze'
    )
    parser.add_argument(
        '--is-sandbox',
        type='bool',
        default=True,
        help='whether the run is a sandbox run or not',
    )
    parser.add_argument(
        '--outdir', type=str, default=None, help='where to save the results'
    )
    parser.add_argument(
        '--pairings-filepath',
        type=str,
        default=None,
        help='path to the acute analysis pairs for the corresponding run id',
    )
    parser.add_argument(
        '--rounding-digit',
        type=int,
        default=2,
        help='number of digits for rounding displayed table',
    )
    parser.add_argument(
        '--max-matchups-html',
        type=int,
        default=10,
        help='max number of matchups to display per model pair in html',
    )
    parser.add_argument(
        '--min-dialogue-length',
        type=int,
        default=-1,
        help='the minimum number of turns for both dialogues in a matchup to be counted as valid for analysis',
    )
    parser.add_argument(
        '--annotate-convo',
        type='bool',
        default=False,
        help='whether to include a checkbox column for annotating the conversation pairs',
    )
    return parser


def _print_progress(msg: str):
    """
    Format a msg to print to stdout well.

    :param msg:
        message to print
    """
    print(f"\n{'-' * 60}\n {msg} \n {'-' * 60}")


class AcuteAnalyzer(object):
    """
    Analyzer.

    Given a run_id, we can do lots of fun things!
    """

    def __init__(self, opt: Dict, remove_failed: bool = True):
        """
        Initialize the analyzer.

        Builds up the dataframe

        :param opt:
            opt dict

        :param remove_failed:
            Whether to remove ratings from turkers who failed onboarding
        """
        self.ROOT_DIR = os.path.join(opt['datapath'], 'acute_eval')
        self.run_id = opt['run_id']
        self.is_sandbox = opt['is_sandbox']
        self.outdir = opt['outdir']
        self.pairings_filepath = opt['pairings_filepath']
        if not self.pairings_filepath:
            self.pairings_filepath = ''
        if not self.outdir:
            self.outdir = os.path.join(self.ROOT_DIR, f'{self.run_id}-results')
        if not os.path.exists(self.ROOT_DIR):
            os.mkdir(self.ROOT_DIR)
        if not os.path.exists(self.outdir):
            os.mkdir(self.outdir)
        self.db_path = os.path.join(
            os.path.dirname(parlai_filepath),
            'mturk',
            'run_data',
            f"pmt_{'sb' if opt['is_sandbox'] else ''}data.db",
        )
        self.dataframe = self._extract_to_dataframe()
        self.min_dialogue_length = opt.get('min_dialogue_length', -1)
        self.max_matchups_html = opt.get('max_matchups_html', 10)
        self.annotate_convo = opt.get('annotate_convo', False)
        if remove_failed:
            self._remove_failed_onboarding()
        self._extract_model_names()
        self._load_pairings_file()

    def _extract_response_data(
        self,
        data: Dict[str, Any],
        task_data: Dict[str, Any],
        hit: Dict[str, Any],
        response_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Extract response data from task data.

        :param data:
            full data from a given turker
        :param task_data:
            data from one "task" completion (i.e. one dialogue comparison)
        :param hit:
            hit data
        :param response_data:
            turker's corresponding response data corresponding to the task

        :return response:
            Turker's response data from the task
        """
        response: Dict[str, Any] = {
            'run_id': self.run_id,
            'worker': data['worker_id'],
            'time_taken': hit['task_end'] - hit['task_start'],
            'question': data['task_data'][0]['task_specs']['question'],
            'conversation_id': hit['conversation_id'],
        }
        onboarding = task_data['task_specs'].get('is_onboarding', False)
        choice = response_data['speakerChoice']
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
                'reason': response_data['textReason'],
                'is_onboarding': onboarding,
                'matchup': f"{'__vs__'.join(speakers_to_eval)}",
                'pairing_id': task_data['pair_id'],
                'dialogue_lengths': {
                    task_data['task_specs']['model_left']['name']: len(
                        task_data['task_specs']['model_left']['dialogue']
                    ),
                    task_data['task_specs']['model_right']['name']: len(
                        task_data['task_specs']['model_right']['dialogue']
                    ),
                },
                'speaker_model_mapping': [
                    task_data['task_specs']['model_left']['name'],
                    task_data['task_specs']['model_right']['name'],
                ],
            }
        )
        return response

    def _get_hit_data(
        self, hit: Dict[str, Any], logger: MTurkDataHandler
    ) -> Optional[Dict[str, Any]]:
        """
        Return data for a given hit.

        If the HIT is corrupt for whatever reason, we return None

        :param hit:
            HIT information dict
        :param logger:
            Data handler

        :return data:
            Optional dict with the hit data
        """
        try:
            full_data: Dict[str, Any] = logger.get_full_conversation_data(
                self.run_id, hit['conversation_id'], self.is_sandbox
            )
        except FileNotFoundError:
            print(
                f"WARNING: Data for run_id `{self.run_id}` not found for "
                f"conversation id {hit['conversation_id']}"
            )
            return None

        data: Dict[str, Any] = next(iter(full_data['worker_data'].values()))
        if not (
            'task_data' in data['response'] and len(data['response']['task_data']) > 0
        ):
            # worker abandoned task, drop their annotations
            return None
        elif len(data['response']['task_data']) != len(data['task_data']):
            raise ValueError('Saved task data does not match response task data')

        return data

    def _extract_to_dataframe(self) -> pd.DataFrame:
        """
        Extract the data from the run to a pandas dataframe.
        """
        logger = MTurkDataHandler(file_name=self.db_path)
        hits = logger.get_pairings_for_run(self.run_id)

        dataframe: List[Dict[str, Any]] = []
        for hit in hits:
            if hit['conversation_id'] is None:
                continue
            data = self._get_hit_data(hit, logger)
            if data is None:
                continue
            for r_idx, task_data in enumerate(data['task_data']):
                response_data = data['response']['task_data'][r_idx]
                if response_data is None:
                    continue
                response = self._extract_response_data(
                    data, task_data, hit, response_data
                )
                dataframe.append(response)
        return pd.DataFrame(dataframe)

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

    def _load_pairings_file(self):
        """
        Load the pairings file, if provided.

        Allows for visualization of the conversations turkers rated.
        """
        df = self.dataframe
        if not self.pairings_filepath or not os.path.exists(self.pairings_filepath):
            return
        pairings = []
        with open(self.pairings_filepath, 'r') as f:
            for line in f:
                pair = json.loads(line)
                model1, model2 = pair['speakers_to_eval']
                pair[model1] = pair['dialogue_dicts'][0]
                pair[model2] = pair['dialogue_dicts'][1]
                del pair['dialogue_dicts']
                pairings.append(pair)
        pairs_to_eval = [pairings[i] for i in df.pairing_id.values.tolist()]
        # Build dialogue_ids => dialogue mappings

        winner_dialogues = []
        loser_dialogues = []
        for i, (_, row) in enumerate(df.iterrows()):
            winner = row['winner']
            loser = row['loser']
            winner_dialogues.append(pairs_to_eval[i][winner])
            loser_dialogues.append(pairs_to_eval[i][loser])
        df.loc[:, 'pairs_to_eval'] = pd.Series(pairs_to_eval, index=df.index)
        df.loc[:, 'winner_dialogue'] = pd.Series(winner_dialogues, index=df.index)
        df.loc[:, 'loser_dialogue'] = pd.Series(loser_dialogues, index=df.index)
        self.dataframe = df

    def _extract_model_names(self):
        """
        Extract the model nicknames from the dataframe.
        """
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

    ###############################
    # Data manipulation Functions #
    ###############################
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

    def filter_by_dialogue_length(self, is_debug=False):
        """
        Filter out matchup with one of the conversation shorter than
        self.min_dialogue_length This applies to calculating sorted_win_frac_df and
        signficance_df, but not html visualizations of conversations.

        :param is_debug: if True, print logs indicating the number of pairings filtered out due to short conversation.
            is_debug bool
        """
        df = pd.DataFrame()
        filter_list = {}
        for _, row in self.dataframe.iterrows():
            keep_row = True
            for model_name, dialogue_length in row['dialogue_lengths'].items():
                if keep_row and dialogue_length < self.min_dialogue_length:
                    keep_row = False
                    filter_list[model_name] = filter_list.get(model_name, 0) + 1
            if keep_row:
                df = df.append(row, ignore_index=True)
        if is_debug:
            for model_name in filter_list:
                print(
                    f"For {self.run_id}: filter out {filter_list[model_name]} matchups due to {model_name} with dialogue length shorter than {self.min_dialogue_length}"
                )
        return df

    def get_wins_per_model_matchup(self) -> pd.DataFrame:
        """
        Return the wins for each model by matchup.
        """
        df_filtered = self.filter_by_dialogue_length(True)
        self.matchup_total_df = (
            df_filtered.groupby(['eval_choice_0', 'eval_choice_1'])['run_id']
            .count()
            .to_frame('matchup_total')
        )
        self.win_total_df = (
            df_filtered.groupby(['eval_choice_0', 'eval_choice_1', 'winner', 'loser'])[
                'loser'
            ]
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
        self.models_by_win_frac = (
            self.win_fraction_df.groupby("winner")["win_frac"]
            .mean()
            .sort_values()
            .index.values.tolist()
        )
        self.sorted_win_frac_df = pivoted_df.reindex(
            index=self.models_by_win_frac, columns=self.models_by_win_frac
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
            .reindex(index=self.models_by_win_frac, columns=self.models_by_win_frac)
        )
        return self.num_hits_per_matchup_df

    ##################
    # Rendering HTML #
    ##################
    def render_conversations_per_matchups(self):
        """
        Render conversations with and without reasons included.
        """
        matchups = list(self.dataframe.matchup.unique())

        def _render_row(matchup: List[str], row: pd.Series, row_id: int) -> str:
            dialogues = {'winner_dialogue': '', 'loser_dialogue': ''}
            for d_key in dialogues:
                result = []
                for _, turn in enumerate(row[d_key]['dialogue']):
                    speakername = turn['id']
                    text = turn['text']
                    is_bot = (
                        (speakername != 'human_evaluator')
                        and (speakername != 'other_speaker')
                        and speakername in matchup
                    )
                    align = 'right' if is_bot else 'left'
                    color = "white" if is_bot else "black"
                    bgcolor = '#2391f7' if is_bot else '#e1e1e7'

                    result.append(
                        (
                            '<div style="overflow: auto; padding: 1ex 0;">'
                            f'<div style="clear: both; float: {align}; color: {color}; background-color: {bgcolor}; padding: 0.5em 1em; border-radius: 1em; max-width: 80%">'
                            f'<p style="margin: 0">{speakername}: {text}</p>'
                            '</div>'
                            '</div>'
                        )
                    )
                dialogues[d_key] = (
                    '<div style="background-color: white; margin: 0em; padding: 0.5em; '
                    'font-family: sans-serif; font-size: 9pt; width: 99%;">'
                    + ''.join(result)
                    + '</div>'
                )

            if row['winner'] == row['speaker_model_mapping'][0]:
                speakers_footnote = "(Speaker_1[winner] = {}, Speaker_2 = {})".format(
                    row['speaker_model_mapping'][0], row['speaker_model_mapping'][1]
                )
            else:
                speakers_footnote = "(Speaker_1 = {}, Speaker_2[winner] = {})".format(
                    row['speaker_model_mapping'][0], row['speaker_model_mapping'][1]
                )

            checkbox_row = (
                '<td>'
                '<div><input type= "checkbox" id= "cherry" name= "cherry">'
                '<label for="cherry">Cherry</label>'
                '</div>'
                '<div><input type= "checkbox" id= "lemon" name= "lemon">'
                '<label for= "lemon">Lemon</label>'
                '</div>'
                '<div><input type= "checkbox" id= "neutral" name= "neutral">'
                '<label for= "neutral">Neutral</label>'
                '</div>'
                '</td>'
            )
            dialogue_row = f"<td>{dialogues['winner_dialogue']}</td><td>{dialogues['loser_dialogue']}</td>"
            reason_row = f"<td>{row['reason']}\n{speakers_footnote}</td>"
            if self.annotate_convo:
                return f"<tr><td>Pair {str(row_id)}</td>{checkbox_row}{dialogue_row}{reason_row}</tr>"
            else:
                return f"<tr><td>Pair {str(row_id)}</td>{dialogue_row}{reason_row}</tr>"

        def _render_html(table: pd.DataFrame) -> HTML:
            result = '\
                <div id="toc_container">\
                <p class="toc_title">Model Pairs</p>\
                <ul class="toc_list">'
            for matchup in matchups:
                eval_question = table.loc[table['matchup'] == matchup, 'question'].iloc[
                    0
                ]
                result += f"<li><a href='#{matchup}''>{matchup + '__on__' +eval_question}</a></li>"
            result += '</ul></div>'
            for matchup in matchups:
                length = min(
                    self.max_matchups_html, len(table[table['matchup'] == matchup])
                )
                eval_question = table.loc[table['matchup'] == matchup, 'question'].iloc[
                    0
                ]
                matchup_table = table[table['matchup'] == matchup][:length]
                table_rows = [
                    _render_row(matchup.split('__vs__'), row, idx)
                    for idx, (_, row) in enumerate(matchup_table.iterrows())
                ]
                if self.annotate_convo:
                    table_body = f"<table border=1 frame=void rules=rows cellpadding='20'><tr><th>Pair Id</th><th>Comments</th><th>Winner Conversation</th><th>Loser Conversation</th><th>Reason</th></tr>{''.join(table_rows)}</table>"
                else:
                    table_body = f"<table border=1 frame=void rules=rows cellpadding='20'><tr><th>Pair Id</th><th>Winner Conversation</th><th>Loser Conversation</th><th>Reason</th></tr>{''.join(table_rows)}</table>"
                result += f"<h2 id='{matchup}'><li><a href='#toc_container'>{matchup + '__on__' +eval_question}</a></li></h2><body>{table_body}</body>"
            return HTML(result)

        table = self.dataframe

        self.rendered_without_reasons = _render_html(table)
        table = table[table['reason'] != '']
        self.rendered_with_reasons = _render_html(table)

    #############################
    # Functionality from Notebook
    #############################
    def get_matchup_totals_with_signficance(self) -> pd.DataFrame:
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
        df_filtered = self.filter_by_dialogue_length()
        for _, run_annotations in df_filtered.groupby('run_id'):
            question = list(run_annotations.question)[0]
            for matchup, annotations in run_annotations.groupby('matchup'):
                model1, model2 = matchup.split('__vs__')
                wincount1 = np.sum(annotations['winner'] == model1)
                wincount2 = np.sum(annotations['winner'] == model2)
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
                        'p': p,
                        'stars': stars,
                        'sigf': plevel,
                        'agree': total_agreement,
                    }
                )
        df_output = pd.DataFrame(output)
        # order the columns how we want
        self.signficance_df = df_output[
            [
                'question',
                'matchup',
                'model1',
                'numwins1',
                'winrate1',
                'model2',
                'numwins2',
                'winrate2',
                'sigf',
                'stars',
                'p',
                'agree',
            ]
        ]
        return self.signficance_df

    def save_results(self, path: str = None):
        """
        Save results to a certain path.

        This function saves the dataframe results to a csv.

        In addition, if the pairings filepath is specified, this function will also
        save rendered conversations to an html file for viewing.
        """
        if not hasattr(self, 'signficance_df'):
            self.get_matchup_totals_with_signficance()
        if path is None:
            path = self.outdir

        def _path(filename):
            return os.path.join(path, f"{self.run_id}.{filename}")

        # Save significance file
        with open(_path('significance.csv'), 'w') as f:
            f.write(self.signficance_df.to_csv(index=False))
        _print_progress(
            f"To visualize signficance result, try cat {_path('significance.csv')} | column -t -s, | less -S"
        )
        # Save Win Grid
        with open(_path('grid.csv'), 'w') as f:
            f.write(self.get_win_fractions().to_csv(index=True))
        _print_progress(
            f"To visualize grid result, try cat {_path('grid.csv')} | sed 's/,/ ,/g' | column -t -s, | less -S"
        )

        # Render conversations if valid pairings filepath provided
        if os.path.exists(self.pairings_filepath):
            if not hasattr(self, 'rendered_with_reasons'):
                self.render_conversations_per_matchups()
            with open(_path('reason.html'), 'w') as f:
                f.write(self.rendered_with_reasons.data)
            _print_progress(
                f"To visualize only conversations with reasons provided, open {_path('reason.html')} in a browser"
            )
            with open(_path('all.html'), 'w') as f:
                f.write(self.rendered_without_reasons.data)
            _print_progress(
                f"To visualize all conversations, open {_path('all.html')} in a browser"
            )


if __name__ == "__main__":
    parser = setup_args()
    args = parser.parse_args()
    analyzer = AcuteAnalyzer(args)
    results = pd.DataFrame(analyzer.get_win_fractions())
    analyzer.save_results()
    _print_progress('Here is the grid of results:')
    rounding_digit = args.get('rounding_digit', 2)
    print(results.round(rounding_digit).to_string())
    result = pd.DataFrame(analyzer.get_matchup_totals_with_signficance())
    result = result.drop(columns=['agree'])
    _print_progress('Here is each matchup with signficance measures:')
    print(result.round(rounding_digit).to_string())
