#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from parlai.crowdsourcing.utils.analysis import AbstractTurnAnnotationResultsCompiler


class TurnAnnotationsStaticResultsCompiler(AbstractTurnAnnotationResultsCompiler):
    """
    Class to compile results from static turn annotations.

    Change PROBLEM_BUCKETS in task_config/annotations_config.json to be the buckets that
    you are asking crowdsource workers to annotate with.

    NOTE this script directly accesses Mephisto files rather than using the DataBrowser.
    This makes it fragile and not a great example for extension. It is a good candidate
    for refactor.
    """

    NUM_SUBTASKS = 7
    LIVE_ONBOARDING_IS_LAST_SUBTASK = True
    LIVE_ONBOARDING_THRESHOLD = 0.5
    INFLIGHT_ONBOARDING_DATA = None
    NUM_ANNOTATIONS = 5

    FILENAME_STUB = 'results'
    CALCULATE_STATS_INTERANNOTATOR_AGREEMENT = True

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--results-folders', type=str, help='Comma-separated list of result folders'
        )
        parser.add_argument(
            '--onboarding-in-flight-data-file',
            type=str,
            help='Path to JSONL file containing onboarding in-flight conversations',
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

        if 'results_folders' in opt:
            self.results_folders = opt['results_folders'].split(',')
        else:
            self.results_folders = None

        # Validate problem buckets
        if self.use_problem_buckets and 'none_all_good' not in self.problem_buckets:
            # The code relies on a catchall "none" category if the user selects no other
            # annotation bucket
            raise ValueError(
                'There must be a "none_all_good" category in self.problem_buckets!'
            )
        self.onboarding_in_flight_data_file = opt.get('onboarding_in_flight_data_file')
        self.gold_annotations_file = opt.get('gold_annotations_file')
        if not self.use_problem_buckets:
            raise ValueError(
                'Problem buckets must be used when analyzing results from the static turn annotations task!'
            )

    def get_data_paths_mephisto(self, task_run_id_folder):
        """
        Get all the individual folders with data from the <task_run_id> path we are
        given as input.

        In Mephisto the structure is:
        /<project_id>/<task_run_id>/<assignment_id>/<agent_id>/

        Side note: assignment_id == HIT ID
        """
        # TODO: replace direct folder access with a call to
        #  mephisto.tools.data_browser.DataBrowser
        read_folders = []
        for assignment_id in os.listdir(task_run_id_folder):
            if assignment_id in ['onboarding', 'reservations', 'build', '.', '..']:
                continue
            assignment_folder = os.path.join(task_run_id_folder, assignment_id)
            if os.path.isdir(assignment_folder):
                if len(os.listdir(assignment_folder)) > 2:
                    print(
                        f'Had more than one HIT in folder: {assignment_folder}, had {len(os.listdir(assignment_folder))} folders.'
                    )
                for agent_id in os.listdir(assignment_folder):
                    if os.path.isdir(os.path.join(assignment_folder, agent_id)):
                        full_path = os.path.join(
                            task_run_id_folder,
                            assignment_id,
                            agent_id,
                            'agent_data.json',
                        )
                        read_folders.append(full_path)
        return read_folders

    def get_results_path_base(self) -> str:
        now = datetime.now()
        return os.path.join(
            self.output_folder, f'{self.FILENAME_STUB}_{now.strftime("%Y%m%d_%H%M%S")}'
        )

    def compile_results(self) -> pd.DataFrame:
        # Loads data from files and gets rid of incomplete or malformed convos
        conversations = self.compile_initial_results(self.results_folders)
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

        return main_dataframe

    def _validate_hit(self, hit_data) -> Tuple[bool, Optional[str]]:
        """
        Validate an entire HIT.

        :return: tuple (is_valid, reason)
        """
        if 'outputs' not in hit_data or hit_data['outputs'] is None:
            return False, 'Malformed HIT'

        subtasks = hit_data['outputs']['final_data']
        if len(subtasks) != self.NUM_SUBTASKS:
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

    def compile_initial_results(self, results_folders) -> list:
        """
        Do initial loading and processing of crowdsource data Loads data from all the
        worker ID files and gets rid of incomplete or malformed convos.

        Also adds fields such as worker_id, assignment_id, etc for convenience
        :return: list of JSON objects which represent a conversation with
        annotations d["data"] of each has an array of utterance level data
        """
        print('Starting compile_initial_results...')
        all_data_paths = []
        for f in results_folders:
            # Each one is a HIT completed by a given worker (so if
            # units-per-assignment > 1), then will include the same conversations
            # multiple times annotated by different workers
            data_paths = self.get_data_paths_mephisto(f)
            all_data_paths.extend(data_paths)
        print(f'Got {len(all_data_paths)} folders to read.')

        conversations = []
        task_completion_times = []
        for dp in all_data_paths:
            # Read in file
            with open(os.path.join(dp), 'rb') as f:
                data = json.load(f)

            worker_id = dp.split('/')[-2]
            hit_id = dp.split('/')[-3]
            _ = dp.split('/')[-4]  # Task run

            (is_valid_hit, reason) = self._validate_hit(data)
            if not is_valid_hit:
                print(
                    f'Skipping invalid HIT {hit_id}, worker_id: {worker_id} for reason: {reason}.'
                )
                continue

            # HIT-level metric of HIT completion time has to be done here for now
            try:
                task_completion_time_seconds = (
                    data['times']['task_end'] - data['times']['task_start']
                )
            except KeyError:
                # We've been relying on non-mephisto API, and in 1.0.2 'times' was deprecated
                # so this uses the new location
                metadata_path = os.path.join(os.path.dirname(dp), 'agent_meta.json')
                with open(metadata_path, 'rb') as f:
                    metadata = json.load(f)

                # HIT-level metric of HIT completion time has to be done here for now
                task_completion_time_seconds = (
                    metadata['task_end'] - metadata['task_start']
                )
            print(task_completion_time_seconds)

            subtasks = data['outputs']['final_data']
            if self.LIVE_ONBOARDING_IS_LAST_SUBTASK:
                qc_success_pct = self._get_inflight_onboarding_success_from_subtask(
                    subtasks[-1]
                )
            else:
                qc_success_pct = 0.0
            for subtask_idx, d in enumerate(subtasks):
                if (
                    subtask_idx == (len(subtasks) - 1)
                    and self.LIVE_ONBOARDING_IS_LAST_SUBTASK
                ):
                    # Last subtask is inflight onboarding; don't include it
                    continue
                subtask_data = copy.deepcopy(d)
                # Structure of each subtask is {'subtask_index': XX, 'data': [...]}
                (is_valid_subtask, reason) = self._validate_subtask(d['data'])
                if not is_valid_subtask:
                    print(
                        f'Skipping invalid subtask within HIT: {hit_id}, worker_id: {worker_id} for reason: {reason}.'
                    )
                    continue

                subtask_data['worker_id'] = worker_id
                subtask_data['hit_id'] = hit_id
                subtask_data['folder'] = dp
                subtask_data['subtask_idx'] = subtask_idx
                experimental_design = 'self_chat'
                subtask_data['model_nickname'] = experimental_design + '/' + 'TODO'
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
                    'annotation_id': f'{convo["hit_id"]}_{convo["subtask_idx"]}_{turn_idx}_{convo["worker_id"]}',
                    'conversation_id': f'{convo["hit_id"]}_{convo["subtask_idx"]}',
                    'utterance_id': f'{convo["hit_id"]}_{convo["subtask_idx"]}_{turn_idx}',
                    'turn_idx': turn_idx,
                    'agent_idx': utt['agent_idx'],
                    'folder': convo['folder'],
                    'worker_id': convo['worker_id'],
                    'hit_id': convo['hit_id'],
                    'model_nickname': convo['model_nickname'],
                    'qc_success_pct': convo['qc_success_pct'],
                    'text': utt['text'],
                }
                row = self._add_additional_columns(row=row, utt=utt)
                for k in self.problem_buckets:
                    row[k] = utt[k] if utt['agent_idx'] == 1 else ''
                rows.append(row)
        df = pd.DataFrame(rows)
        print(f'Returning dataframe with {len(df)} annotations.')
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

        bot_only_df = self._problem_bucket_specific_filtering(bot_only_df)

        # Group at the utterance level (summing across workers)
        bot_only_df = bot_only_df.replace(True, 1)
        bot_only_df = bot_only_df.replace(False, 0)
        all_bot_annotations_count = len(bot_only_df)

        # Remove utterances that don't have self.NUM_ANNOTATIONS annotations
        counted_df = bot_only_df.groupby(['utterance_id']).count()
        counted_df = counted_df[counted_df == self.NUM_ANNOTATIONS].dropna()
        bot_only_df = bot_only_df[bot_only_df['utterance_id'].isin(counted_df.index)]
        print(
            f'Removed {all_bot_annotations_count - len(bot_only_df)} that did not have annotations by {self.NUM_ANNOTATIONS} workers. {len(bot_only_df)} annotations remaining.'
        )

        if self.LIVE_ONBOARDING_IS_LAST_SUBTASK:
            # Remove those that didn't get enough right on live onboarding
            bot_only_df = bot_only_df[
                bot_only_df['qc_success_pct'] >= self.LIVE_ONBOARDING_THRESHOLD
            ]

        summed_df = bot_only_df.groupby(['utterance_id']).sum()
        print(f'summed_df has length {len(summed_df)}; bot_only_df: {len(bot_only_df)}')

        utterance_ids = bot_only_df['utterance_id'].unique()
        print(f'Number of unique utterance_ids: {len(utterance_ids)}.')

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
                    kappa_df, [True, False], self.NUM_ANNOTATIONS
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
