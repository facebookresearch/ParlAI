#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from mephisto.data_model.worker import Worker

import parlai.utils.logging as logging
from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import (
    BLUEPRINT_TYPE,
)  # For registering the blueprint
from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.analysis import AbstractResultsCompiler

_ = BLUEPRINT_TYPE
# NOTE: BLUEPRINT_TYPE needs to be imported here to register the blueprint


class ModelChatResultsCompiler(AbstractResultsCompiler):
    """
    Compile and save results of human+model chats.

    Results will be saved on the level of specific conversations, as well as aggregated
    up to the level of each worker as a whole.
    """

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--filter-uniform-hits',
            action='store_true',
            help='Filter out any HITs in which the worker\'s annotations were the exact same on each turn of the conversation',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        super().__init__(opt)

        # Input args
        os.makedirs(self.output_folder, exist_ok=True)
        # TODO: see if this can be moved to the superclass
        self.filter_uniform_hits = opt['filter_uniform_hits']

        # Save paths
        self.unacceptable_worker_ids_path = os.path.join(
            self.output_folder, 'unacceptable_worker_ids.txt'
        )
        self.annotation_selection_rate_path = os.path.join(
            self.output_folder, 'annotation_selection_rates.csv'
        )
        self.likert_score_stat_path = os.path.join(
            self.output_folder, 'likert_score_stats.csv'
        )

        self.acceptability_checker = AcceptabilityChecker()

    def get_results_path_base(self) -> str:
        return os.path.join(self.output_folder, 'results')
        # TODO: see if this can be moved to the superclass

    def compile_results(self) -> pd.DataFrame:

        # Load task data
        logging.info('Retrieving task data from Mephisto.')
        task_units_data = self.get_task_data()
        logging.info(f'Data for {len(task_units_data)} units loaded successfully.')

        num_convos_with_no_save_data = 0
        num_wrong_status_convos = 0
        num_complete_convos = 0

        unacceptable_task_units = []
        unacceptable_worker_ids = []
        conversation_idx = 0
        conversation_dfs = []

        for task_unit in task_units_data:

            worker_id = task_unit['worker_id']
            assignment_id = task_unit['assignment_id']

            # Skipping this conversation if save data is not found or the status is
            # invalid
            if task_unit['data']['save_data'] is None:
                logging.info('Found a task unit with no save data! Skipping.')
                num_convos_with_no_save_data += 1
                continue
            elif task_unit['status'] not in ['completed', 'approved']:
                logging.info(
                    f'Found a HIT with the status "{task_unit["status"]}"!.'
                    f'Skipping.'
                )
                num_wrong_status_convos += 1
                continue
            else:
                num_complete_convos += 1

            # Extract out useful conversation-level data
            custom_data = task_unit['data']['save_data']['custom_data']
            mturk_worker_id = Worker.get(self.get_mephisto_db(), worker_id).worker_name
            task_start = datetime.utcfromtimestamp(task_unit['task_start'])
            task_end = datetime.utcfromtimestamp(task_unit['task_end'])
            info_dict = {
                ('worker_id', ''): worker_id,
                ('mturk_worker_id', ''): mturk_worker_id,
                ('unit_id', ''): task_unit['unit_id'],
                ('assignment_id', ''): assignment_id,
                ('conversation_idx', ''): conversation_idx,
                ('date', ''): task_start.strftime('%Y-%m-%d'),
                ('completion_time', ''): (task_end - task_start).total_seconds(),
            }

            # Check that the conversation consists of pairs of comments between
            # Speaker 1 and Speaker 2, with Speaker 1 speaking first
            assert 'final_rating' in task_unit['data']['messages'][-1]['task_data']
            convo_messages = [m for m in task_unit['data']['messages'][:-1]]
            # The final message is just a final rating
            assert all(
                [
                    message['id'] == 'Speaker 2' if message_idx % 2 else 'Speaker 1'
                    for message_idx, message in enumerate(convo_messages)
                ]
            )
            messages_1 = [m for m in convo_messages if m['id'] == 'Speaker 1']
            messages_2 = [m for m in convo_messages if m['id'] == 'Speaker 2']
            assert len(messages_1) + len(messages_2) == len(convo_messages)

            # Determine whether the HIT contains unacceptable messages. (We do this for
            # every HIT, even if acceptability violation info was already saved, because
            # the violation criteria may have changed since the HIT was collected.)
            utterances_1 = [m['text'] for m in messages_1]
            assert utterances_1[0] == 'Hi!', (
                'This script assumes that the first human message is "Hi!", which is '
                'set by default and cannot be changed by the crowdsourcing worker.'
            )
            acceptability_violations = self.acceptability_checker.check_messages(
                messages=utterances_1[1:],  # Don't use the initial "Hi!"
                is_worker_0=True,
                violation_types=self.acceptability_checker.ALL_VIOLATION_TYPES,
            )
            # Here, "worker 0" refers to Speaker 1, because we mix 0- and 1-indexing
            if acceptability_violations != '':
                logging.info(
                    f'Conversation fails acceptability checks with a violation of '
                    f'"{acceptability_violations}", given the following utterances: '
                    f'{utterances_1[1:]}. Skipping.'
                )
                unacceptable_task_units.append(task_unit)
                assert (
                    mturk_worker_id is not None
                ), "MTurk worker ID cannot be determined for this unacceptable conversation!"
                unacceptable_worker_ids.append(mturk_worker_id)
                continue

            # Ignore the conversation if ratings for all turns are the same, because
            # it's somewhat implausible that *all* turns in a conversation should garner
            # the same rating of engagingness, humanness, interestingness, or none.
            # (However, don't put these workers on the "unacceptable worker IDs" list,
            # to give them a little bit of the benefit of the doubt: i.e. maybe the
            # worker just didn't try hard enough to find which responses were more
            # engaging, etc. than others, but that doesn't mean that all of their HITs
            # across all evals are bad and should be removed.)
            if self.filter_uniform_hits:
                annotations = [
                    m['task_data']['problem_data_for_prior_message']
                    for m in task_unit['data']['messages']
                    if 'problem_data_for_prior_message' in m.get('task_data', {})
                ]
                hashable_annotations = [
                    tuple(a[key] for key in sorted(a.keys())) for a in annotations
                ]
                unique_annotations = set(hashable_annotations)
                if len(unique_annotations) < 1:
                    raise ValueError('No annotations found for this HIT!')
                elif len(unique_annotations) == 1:
                    logging.info(
                        f'All model responses in the conversation received the same '
                        f'annotation: {hashable_annotations[0]}. Skipping.'
                    )
                    unacceptable_task_units.append(task_unit)
                    continue

            single_turn_dicts = []

            # Compile personas and previous utterances
            text_parts = []
            if custom_data['personas'] is not None and len(custom_data['personas']) > 0:
                assert len(custom_data['personas']) == 2
                text_parts += [
                    'HUMAN PERSONA: ' + ' '.join(custom_data['personas'][0]),
                    'BOT PERSONA: ' + ' '.join(custom_data['personas'][1]),
                ]
            if (
                custom_data['additional_context'] is not None
                and len(custom_data['additional_context']) > 0
            ):
                text_parts.append(
                    'ADDITIONAL CONTEXT: ' + custom_data['additional_context']
                )
            single_turn_dicts.append(
                {**info_dict, ('context', ''): ' '.join(text_parts)}
            )

            # Loop over conversation turns
            turns_per_speaker = defaultdict(int)
            for message in task_unit['data']['messages']:
                if 'text' in message:

                    speaker_id = message['id']

                    # Add in annotation results, if they exist
                    if 'problem_data_for_prior_message' in message.get('task_data', {}):
                        bucket_data = {
                            ('annotation_bucket', bucket): value
                            for bucket, value in message['task_data'][
                                'problem_data_for_prior_message'
                            ].items()
                        }
                    else:
                        bucket_data = {}

                    # Add in results from the final rating(s), if they exist
                    if 'final_rating' in message.get('task_data', {}):
                        ratings = message['task_data']['final_rating'].split('|')
                        final_rating_data = {
                            ('final_rating', str(idx)): value
                            for idx, value in enumerate(ratings)
                        }
                    else:
                        final_rating_data = {}

                    turns_per_speaker[speaker_id] += 1

                    single_turn_dicts.append(
                        {
                            **info_dict,
                            ('speaker_id', ''): speaker_id,
                            ('speaker_turn_idx', ''): turns_per_speaker[speaker_id],
                            ('text', ''): message['text'].replace('\n', '__newline__'),
                            **bucket_data,
                            **final_rating_data,
                        }
                    )

            # Adding the full conversation to the list of conversations
            single_turn_series = [
                pd.Series(dict_).to_frame().transpose() for dict_ in single_turn_dicts
            ]
            single_convo_df = pd.concat(single_turn_series, axis=0, sort=False)
            conversation_dfs.append(single_convo_df)
            conversation_idx += 1

        logging.info(
            f'{num_convos_with_no_save_data:d} conversations found with no save data.'
        )
        logging.info(
            f'{num_wrong_status_convos:d} conversations found with the wrong status.'
        )
        logging.info(f'{num_complete_convos:d} complete conversations found:')
        logging.info(f'\t{len(unacceptable_task_units):d} unacceptable conversations.')
        logging.info(f'\t{len(conversation_dfs):d} acceptable conversations.')

        # # Compile results across all conversations

        if len(conversation_dfs) == 0:
            raise ValueError('No acceptable conversations found!')
        unordered_conversation_df = pd.concat(conversation_dfs, axis=0)
        initial_ordered_columns = list(info_dict.keys()) + [
            ('context', ''),
            ('speaker_id', ''),
            ('speaker_turn_idx', ''),
            ('text', ''),
        ]
        all_ordered_columns = initial_ordered_columns + [
            col
            for col in unordered_conversation_df.columns
            if col not in initial_ordered_columns
        ]
        conversation_df = unordered_conversation_df[all_ordered_columns]
        # TODO: is there a less hacky way than this, which relies on the most recent
        #  value of `info_dict`, to put the columns back into the right order?

        # # Calculate and save auxiliary stats

        logging.info(
            f'Saving MTurk IDs of workers with unacceptable conversations to '
            f'{self.unacceptable_worker_ids_path}.'
        )
        with open(self.unacceptable_worker_ids_path, 'w') as f:
            for worker_id in unacceptable_worker_ids:
                f.write(worker_id + '\n')

        # Calculate rates of selecting various annotation buckets
        annotation_bucket_df = conversation_df['annotation_bucket'].dropna(
            axis=0, how='any'
        )
        if annotation_bucket_df.isna().sum().sum() > 0:
            raise ValueError(
                'There is at least one row in which only partial annotation bucket data exists!'
            )
        annotation_selection_rate_df = annotation_bucket_df.mean().to_frame(
            'selection_rate'
        )
        annotation_selection_rate_df.to_csv(self.annotation_selection_rate_path)
        logging.info(
            f'Annotation bucket selection rates saved to {self.annotation_selection_rate_path}.'
        )
        output_strings = [
            f'{series.name}: {100*series["selection_rate"]:0.0f}%'
            for _, series in annotation_selection_rate_df.iterrows()
        ]
        logging.info('Annotation bucket selection rates:\n' + '\n'.join(output_strings))

        # Calculate Likert score stats
        final_rating_df = conversation_df['final_rating'].dropna(axis=0, how='any')
        if final_rating_df.isna().sum().sum() > 0:
            raise ValueError(
                'There is at least one row in which only partial final rating data exists!'
            )
        likert_score_stat_df = final_rating_df.astype(int).describe()
        likert_score_stat_df.to_csv(self.likert_score_stat_path)
        logging.info(f'Likert score statistics saved to {self.likert_score_stat_path}.')
        logging.info(f'Mean Likert scores:\n{likert_score_stat_df.loc["mean"]}')

        return conversation_df


if __name__ == '__main__':
    parser_ = ModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    ModelChatResultsCompiler(vars(args_)).compile_and_save_results()
