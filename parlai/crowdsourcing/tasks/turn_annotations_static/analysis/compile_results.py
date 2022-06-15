#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from mephisto.data_model.worker import Worker

from parlai.crowdsourcing.tasks.turn_annotations_static.turn_annotations_blueprint import (
    STATIC_BLUEPRINT_TYPE,
    STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE,
)
from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.analysis import AbstractTurnAnnotationResultsCompiler


# Importing blueprint type strings to force registration of the blueprints; we're not
# using the strings themselves
_ = STATIC_BLUEPRINT_TYPE
_ = STATIC_IN_FLIGHT_QA_BLUEPRINT_TYPE


class TurnAnnotationsStaticResultsCompiler(AbstractTurnAnnotationResultsCompiler):
    """
    Class to compile results from static turn annotations.

    Change PROBLEM_BUCKETS in task_config/annotations_config.json to be the buckets that
    you are asking crowdsource workers to annotate with.
    """

    LIVE_ONBOARDING_THRESHOLD = 0.5
    INFLIGHT_ONBOARDING_DATA = None

    FILENAME_STUB = 'results'
    CALCULATE_STATS_INTERANNOTATOR_AGREEMENT = True

    NONE_STRING = '(none)'  # Indicates the absence of a field

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--num-subtasks',
            type=int,
            default=7,
            help='Number of subtasks run per HIT',
        )
        parser.add_argument(
            '--num-annotations',
            type=int,
            default=5,
            help='Minimum number of annotations required per utterance',
        )
        parser.add_argument(
            '--remove-unacceptable-responses',
            action='store_true',
            help='Check quality of responses and filter out unacceptable ones',
        )
        parser.add_argument(
            '--onboarding-in-flight-data-file',
            type=str,
            default=None,
            help='Path to JSONL file containing onboarding in-flight conversations. Unset if no in-flight conversations.',
        )
        parser.add_argument(
            '--gold-annotations-file',
            type=str,
            default=None,
            help='Path to a JSON file mapping utterance IDs to the gold annotations',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):
        super().__init__(opt)

        self.use_none_all_good = 'none_all_good' in self.problem_buckets
        if not self.use_problem_buckets:
            raise ValueError(
                'Problem buckets must be used when analyzing results from the static turn annotations task!'
            )

        self.num_subtasks = opt['num_subtasks']
        self.num_annotations = opt['num_annotations']
        self.remove_unacceptable_responses = opt['remove_unacceptable_responses']
        self.onboarding_in_flight_data_file = opt['onboarding_in_flight_data_file']
        self.live_onboarding_is_last_subtask = (
            self.onboarding_in_flight_data_file is not None
        )
        self.gold_annotations_file = opt.get('gold_annotations_file')

        # Set up acceptability checking of responses
        self.acceptability_checker = AcceptabilityChecker()
        self.violation_types = ['min_words', 'exact_match']
        # Crowdsourcing workers have been known to give too few words or repeat their
        # reason on every turn
        self.acceptability_violations_warning = 'Acceptability violation(s)'
        # Include this at the start of an acceptability violation warning
        self.unacceptable_workers = set()
        # Will contain all workers delivering unacceptable responses

    def get_results_path_base(self) -> str:
        now = datetime.now()
        return os.path.join(
            self.output_folder, f'{self.FILENAME_STUB}_{now.strftime("%Y%m%d_%H%M%S")}'
        )

    def compile_results(self) -> pd.DataFrame:
        # Loads data from files and gets rid of incomplete or malformed convos
        conversations = self.compile_initial_results()
        main_dataframe = self.process_data_into_dataframe(conversations)
        self.calculate_basic_interannotator_agreement(main_dataframe)
        if self.gold_annotations_file is not None:
            with open(self.gold_annotations_file, 'r') as gold_f:
                gold_annotations = json.loads(gold_f.read())
                self.calculate_agreement_with_gold_annotations(
                    gold_annotations, main_dataframe
                )
        if self.CALCULATE_STATS_INTERANNOTATOR_AGREEMENT:
            self.calculate_stats_interannotator_agreement(main_dataframe)

        if (~main_dataframe['other_metadata'].isna()).sum() > 0:
            metadata_grouped = main_dataframe.groupby('other_metadata').agg(
                {
                    bucket: (lambda sr: sr[~(sr == '')].astype(int).mean())
                    for bucket in self.problem_buckets
                }
            )
            print('\nMean bucket selection rates grouped by metadata value:')
            print(metadata_grouped)
            output_path = self.get_results_path_base() + '.metadata_grouped.csv'
            metadata_grouped.to_csv(output_path)

        if self.remove_unacceptable_responses:
            print(
                f'{len(self.unacceptable_workers):d} workers with unacceptable responses found:'
            )
            for mturk_worker_id in sorted(list(self.unacceptable_workers)):
                print(mturk_worker_id)
        return main_dataframe

    def _validate_hit(self, hit_data) -> Tuple[bool, Optional[str]]:
        """
        Validate an entire HIT.

        :return: tuple (is_valid, reason)
        """
        if 'outputs' not in hit_data or hit_data['outputs'] is None:
            return False, 'Malformed HIT'

        subtasks = hit_data['outputs']['final_data']
        if len(subtasks) != self.num_subtasks:
            return False, f'Incomplete HIT with subtask length {len(subtasks)}.'

        return True, None

    def _validate_subtask(self, subtask_data) -> Tuple[bool, Optional[str]]:
        """
        Validate a conversation subtask within the HIT.

        :return: tuple (is_valid, reason)
        """
        # Check that the conversation consists of pairs of comments between
        # agents 0 and 1, with 0 speaking first
        try:
            assert all(
                [
                    utterance_data['agent_idx'] == turn_idx % 2
                    for turn_idx, utterance_data in enumerate(subtask_data)
                ]
            )
            messages_0 = [utt for utt in subtask_data if utt['agent_idx'] == 0]
            messages_1 = [utt for utt in subtask_data if utt['agent_idx'] == 1]
            assert len(messages_0) + len(messages_1) == len(subtask_data)
        except Exception:
            return False, f'Data not in form expected. Length is: {len(subtask_data)}'

        for utterance_data in subtask_data:
            if (
                utterance_data['agent_idx'] == 1
                and self.problem_buckets[0] not in utterance_data
            ):
                return (
                    False,
                    f'Bot utterance was malformed and had no problem annotation fields (Failed to find key: {self.problem_buckets[0]}).',
                )

        # Check the responses for acceptability
        if self.remove_unacceptable_responses:
            responses = [
                utt['input_response'] for utt in subtask_data if utt['agent_idx'] == 1
            ]
            acceptability_violations_0 = self.acceptability_checker.check_messages(
                messages=responses,
                is_worker_0=False,
                violation_types=self.violation_types,
            )
            # `is_worker_0` only applies to the 'penalize_greetings' violation, which
            # we're not using anyway
            if acceptability_violations_0 == '':
                return True, None
            else:
                return (
                    False,
                    f'{self.acceptability_violations_warning}: {acceptability_violations_0}',
                )
        else:
            return True, None

    def _get_inflight_onboarding_success_from_subtask(self, subtask):
        if self.INFLIGHT_ONBOARDING_DATA is None:
            self.INFLIGHT_ONBOARDING_DATA = self.setup_inflight_onboarding_data()
        onboarding_utterance = subtask['data'][-1]
        num_answers = 0
        num_correct = 0
        num_incorrect = 0
        for d in self.INFLIGHT_ONBOARDING_DATA:
            if d['dialog'][-1][-1]['text'] == onboarding_utterance['text']:
                num_answers = len(d['answers'])
                for pb in self.problem_buckets:
                    if pb in d['answers'] and onboarding_utterance[pb]:
                        num_correct += 1
                    if onboarding_utterance[pb] and pb not in d['answers']:
                        num_incorrect += 1
        return num_correct / num_answers

    def compile_initial_results(self) -> List[dict]:
        """
        Do initial loading and processing of crowdsource data. Loads data from
        DataBrowser and gets rid of incomplete or malformed convos.

        Also adds fields such as worker_id, assignment_id, etc. for convenience
        :return: list of JSON objects which represent a conversation with
        annotations d["data"] of each has an array of utterance level data
        """
        print('Starting compile_initial_results...')
        task_units_data = self.get_task_data()

        conversations = []
        task_completion_times = []
        for task_unit in task_units_data:

            worker_id = task_unit['worker_id']
            assignment_id = task_unit['assignment_id']
            data = task_unit['data']

            (is_valid_hit, reason) = self._validate_hit(data)
            if not is_valid_hit:
                print(
                    f'Skipping invalid HIT {assignment_id}, worker_id: {worker_id} for reason: {reason}.'
                )
                continue

            # HIT-level metric of HIT completion time has to be done here for now
            task_completion_time_seconds = (
                data['times']['task_end'] - data['times']['task_start']
            )
            print(task_completion_time_seconds)

            subtasks = data['outputs']['final_data']
            if self.live_onboarding_is_last_subtask:
                qc_success_pct = self._get_inflight_onboarding_success_from_subtask(
                    subtasks[-1]
                )
            else:
                qc_success_pct = 0.0
            for subtask_idx, d in enumerate(subtasks):
                if (
                    subtask_idx == (len(subtasks) - 1)
                    and self.live_onboarding_is_last_subtask
                ):
                    # Last subtask is inflight onboarding; don't include it
                    continue
                subtask_data = copy.deepcopy(d)
                # Structure of each subtask is {'subtask_index': XX, 'data': [...]}
                (is_valid_subtask, reason) = self._validate_subtask(d['data'])
                if reason is not None and reason.startswith(
                    self.acceptability_violations_warning
                ):
                    mturk_worker_id = Worker.get(
                        self.get_mephisto_db(), worker_id
                    ).worker_name
                    self.unacceptable_workers.add(mturk_worker_id)
                if not is_valid_subtask:
                    print(
                        f'Skipping invalid subtask within HIT: {assignment_id}, worker_id: {worker_id} for reason: {reason}.'
                    )
                    continue

                subtask_data['worker_id'] = worker_id
                subtask_data['assignment_id'] = assignment_id
                subtask_data['subtask_idx'] = subtask_idx
                subtask_data['qc_success_pct'] = qc_success_pct
                conversations.append(subtask_data)

            task_completion_times.append(task_completion_time_seconds)

        if len(task_completion_times) > 0:
            print(
                f'Average task completion time (seconds) was: '
                f'{np.average(task_completion_times):0.1f}'
            )

        if len(conversations) == 0:
            raise ValueError('No conversations found!')

        return conversations

    def process_data_into_dataframe(self, conversations) -> pd.DataFrame:
        """
        Return one big dataframe of all conversations where a row is an utterance and
        its problem annotations.
        """
        print('Starting process_data_into_dataframe...')
        rows = []
        for _, convo in enumerate(conversations):
            for turn_idx, utt in enumerate(convo['data']):
                row = {
                    'annotation_id': f'{convo["assignment_id"]}_{convo["subtask_idx"]}_{turn_idx}_{convo["worker_id"]}',
                    'conversation_id': f'{convo["assignment_id"]}_{convo["subtask_idx"]}',
                    'utterance_id': f'{convo["assignment_id"]}_{convo["subtask_idx"]}_{turn_idx}',
                    'turn_idx': turn_idx,
                    'agent_idx': utt['agent_idx'],
                    'worker_id': convo['worker_id'],
                    'mturk_worker_id': Worker.get(
                        self.get_mephisto_db(), convo['worker_id']
                    ).worker_name,
                    'assignment_id': convo['assignment_id'],
                    'other_metadata': utt.get('other_metadata') or self.NONE_STRING,
                    'qc_success_pct': convo['qc_success_pct'],
                    'text': utt['text'],
                    'response': utt.get('input_response') or self.NONE_STRING,
                }
                row = self._add_additional_columns(row=row, utt=utt)
                for k in self.problem_buckets:
                    row[k] = utt[k] if utt['agent_idx'] == 1 else ''
                rows.append(row)
        df = pd.DataFrame(rows)
        print(f'Returning dataframe with {len(df):d} conversation turns.')
        return df

    def _add_additional_columns(self, row: Dict[str, Any], utt: dict) -> Dict[str, Any]:
        """
        Add additional columns to the results dataframe.

        If you wish to add additional columns to the results dataframe, use the input
        utterance dict `utt` to define new keys in `row`, which will form one row in the
        final results dataframe.
        """
        _ = utt
        # The utt dict isn't used here, but may be used in subclasses.
        return row

    def calculate_basic_interannotator_agreement(self, df: pd.DataFrame) -> None:
        print('Starting calculate_interannotator_agreement...')
        # Drops the human utterances which don't have problem buckets
        bot_only_df = df.replace('', np.nan)
        # Get rid of the None's, which occur if there were no checkboxes (so if
        # last utterance only option selected)
        bot_only_df = bot_only_df.fillna(value=np.nan)
        bot_only_df = bot_only_df.dropna()

        if self.use_none_all_good:
            bot_only_df = self._problem_bucket_specific_filtering(bot_only_df)

        # Group at the utterance level (summing across workers)
        bot_only_df = bot_only_df.replace(True, 1)
        bot_only_df = bot_only_df.replace(False, 0)
        all_bot_annotations_count = len(bot_only_df)

        # Remove utterances that don't have self.num_annotations annotations
        counted_df = bot_only_df.groupby(['utterance_id']).count()
        counted_df = counted_df[counted_df == self.num_annotations].dropna()
        bot_only_df = bot_only_df[bot_only_df['utterance_id'].isin(counted_df.index)]
        print(
            f'Removed {all_bot_annotations_count - len(bot_only_df)} that did not have annotations by {self.num_annotations} workers. {len(bot_only_df)} annotations remaining.'
        )

        if self.live_onboarding_is_last_subtask:
            # Remove those that didn't get enough right on live onboarding
            bot_only_df = bot_only_df[
                bot_only_df['qc_success_pct'] >= self.LIVE_ONBOARDING_THRESHOLD
            ]

        summed_df = bot_only_df.groupby(['utterance_id']).sum()
        print(f'summed_df has length {len(summed_df)}; bot_only_df: {len(bot_only_df)}')

        utterance_ids = bot_only_df['utterance_id'].unique()
        print(f'Number of unique utterance_ids: {len(utterance_ids)}.')

        if len(utterance_ids) > 0:
            if 'any_problem' in summed_df:
                # We've computed a column marking if any problem exists, so include this
                extended_problem_buckets = self.problem_buckets + ['any_problem']
            else:
                extended_problem_buckets = self.problem_buckets
            for k in extended_problem_buckets:
                one_annotator = len(summed_df[summed_df[k] == 1])
                two_annotators = len(summed_df[summed_df[k] == 2])
                three_annotators = len(summed_df[summed_df[k] >= 3])
                total_problem_annotations = (
                    one_annotator + two_annotators + three_annotators
                )
                total_utterances = len(summed_df[k])
                if total_problem_annotations > 0:
                    print(
                        f'Bucket: {k}, total unique problem utterances: {total_problem_annotations} ({total_problem_annotations/total_utterances:.1%} of all), one annotator: {one_annotator} ({one_annotator/total_problem_annotations:.1%}), two_annotators: {two_annotators} ({two_annotators/total_problem_annotations:.1%}), three+ annotators: {three_annotators} ({three_annotators/total_problem_annotations:.1%})'
                    )

    def _problem_bucket_specific_filtering(
        self, bot_only_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter the bot responses given the specific problem buckets being used.
        """

        non_none_problem_buckets = [
            bucket for bucket in self.problem_buckets if bucket != 'none_all_good'
        ]
        assert len(set(non_none_problem_buckets)) + 1 == len(self.problem_buckets)
        # Make sure problem buckets are all unique

        utterance_count_total = len(bot_only_df)

        # Select which columns are consistent
        is_consistent = (
            bot_only_df[non_none_problem_buckets[0]] + bot_only_df['none_all_good']
        ) < 2
        for bucket in non_none_problem_buckets[1:]:
            is_consistent = is_consistent & (
                (bot_only_df[bucket] + bot_only_df['none_all_good']) < 2
            )
        bot_only_df['is_consistent'] = is_consistent
        bot_only_df = bot_only_df[bot_only_df['is_consistent']]

        # If any of the problem buckets have marked True
        any_problem = bot_only_df[non_none_problem_buckets[0]]
        for bucket in non_none_problem_buckets[1:]:
            any_problem = any_problem | bot_only_df[bucket]
        bot_only_df['any_problem'] = any_problem

        print(
            f'Dropped {utterance_count_total - len(bot_only_df)} inconsistently annotated utterances (none_all_good and a problem bucket). Now have {len(bot_only_df)} utterances.'
        )

        return bot_only_df

    def calculate_agreement_with_gold_annotations(
        self, gold_annotations, df: pd.DataFrame
    ) -> None:
        """
        Assume gold_annotations are a dictionary of the form {utterance_id : {bucket_i:
        true/false}} where utterance_id is taken from the compile_initial_results (i.e.
        mephistohitid_subtaskindex_utteranceidx)
        """
        print('Starting calculate_agreement_with_gold_annotations...')
        # Drops the human utterances which don't have problem buckets
        bot_only_df = df.replace('', np.nan)
        # Get rid of the None's, which occur if there were no checkboxes (so if
        # last utterance only option selected)
        bot_only_df = bot_only_df.fillna(value=np.nan)
        bot_only_df = bot_only_df.dropna()

        # Include only utterances that have gold_annotations
        bot_only_df = bot_only_df[
            bot_only_df['utterance_id'].isin(gold_annotations.keys())
        ]

        print(
            f'Got {len(gold_annotations.keys())} utterances with gold annotations. Found {len(bot_only_df)} utterances matching gold annotations from DataFrame.'
        )

        agreement_map = {pb: [] for pb in self.problem_buckets}
        agreement_map_problem_only = {pb: [] for pb in self.problem_buckets}
        problem_counts = {pb: 0 for pb in self.problem_buckets}
        for utterance_id, gold in gold_annotations.items():
            utterance_df = bot_only_df[bot_only_df['utterance_id'] == utterance_id]
            count_workers = len(utterance_df)

            for pb in self.problem_buckets:
                gold_annotation = gold[pb]
                match_count = utterance_df[utterance_df[pb] == gold[pb]].count()[pb]
                a = float(match_count / count_workers)
                agreement_map[pb].append(a)
                if gold_annotation:
                    agreement_map_problem_only[pb].append(a)
                    problem_counts[pb] += 1
        print(
            f'------------------------\nAverage agreement with {len(gold_annotations)} total gold utterances annotated was:'
        )
        for pb in self.problem_buckets:
            print(
                f'{pb}: {np.average(agreement_map[pb]):.1%} ({problem_counts[pb]} gold problem samples)'
            )
        print('------------------------')
        print(
            f'------------------------\nAverage agreement problem samples only with {len(gold_annotations)} total gold utterances annotated was:'
        )
        for pb in self.problem_buckets:
            print(
                f'{pb}: {np.average(agreement_map_problem_only[pb]):.1%} ({problem_counts[pb]} gold problem samples)'
            )
        print('------------------------')

    def calculate_stats_interannotator_agreement(self, df: pd.DataFrame):
        print('Starting calculate_stats_interannotator_agreement...')
        # Get rid of the human utterances (non-annotated)
        bot_only_df = df.replace('', np.nan)
        # Get rid of the None's, which occur if there were no checkboxes (so if
        # last utterance only option selected)
        bot_only_df = bot_only_df.fillna(value=np.nan)
        bot_only_df = bot_only_df.dropna()
        print(f'Calculating agreement on {len(bot_only_df)} annotations.')

        for pb in self.problem_buckets:
            # Expects a df of rater_id, item_id and "data" column
            kappa_df = df[['annotation_id', 'worker_id', 'utterance_id', pb]]
            kappa_df = kappa_df.rename(
                columns={'worker_id': 'rater_id', 'utterance_id': 'item_id', pb: 'data'}
            )
            try:
                fleiss_kappa = self.compute_fleiss_kappa(
                    kappa_df, [True, False], self.num_annotations
                )
            except Exception as exc:
                print(f'Exception calculating Fleiss Kappa: {exc}. Skipping.')
                continue
            print(f'Fleiss\' kappa for {pb} is: {fleiss_kappa:0.3f}.')

    def compute_fleiss_kappa(
        self, df: pd.DataFrame, categories: list, number_of_raters: int
    ) -> float:
        """
        Expects a df of index, rater_id, item_id and "data" column with each row a label
        of one of the categories.
        """

        # categories are "True" and "False"
        # As per wikipedia definition: https://en.wikipedia.org/wiki/Fleiss%27_kappa
        items_list = df.drop_duplicates(subset=['item_id'])['item_id'].to_list()
        N = len(items_list)
        p_j = np.zeros(len(categories))
        P_bar_sum_term = 0.0
        for item in items_list:
            df_item_annotations = df[df['item_id'] == item]
            if len(df_item_annotations) != number_of_raters:
                continue
            for j, c in enumerate(categories):
                try:
                    n_ij = df_item_annotations['data'].value_counts()[c]
                except Exception:
                    n_ij = 0.0
                p_j[j] += n_ij
                P_bar_sum_term += n_ij**2

        p_j = [tmp / (N * number_of_raters) for tmp in p_j]
        P_e_bar = sum([tmp**2 for tmp in p_j])

        P_bar = (P_bar_sum_term - N * number_of_raters) / (
            N * number_of_raters * (number_of_raters - 1)
        )

        kappa = (P_bar - P_e_bar) / (1 - P_e_bar)
        return kappa

    def setup_inflight_onboarding_data(self):
        print('setup_inflight_onboarding_data')
        raw_qc_convos = []
        with open(self.onboarding_in_flight_data_file, "r") as f:
            line = f.readline()
            while line:
                qc_convo = json.loads(line)
                raw_qc_convos.append(qc_convo)
                line = f.readline()
        return raw_qc_convos


if __name__ == '__main__':
    parser_ = TurnAnnotationsStaticResultsCompiler.setup_args()
    args = parser_.parse_args()
    TurnAnnotationsStaticResultsCompiler(vars(args)).compile_and_save_results()
