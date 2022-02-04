#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import re
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict

import pandas as pd

import parlai.utils.logging as logging
from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.analysis import AbstractResultsCompiler
from parlai.crowdsourcing.tasks.pairwise_per_turn_eval.per_turn_eval_blueprint import (
    BLUEPRINT_TYPE,
)  # For registering the blueprint

_ = BLUEPRINT_TYPE
# NOTE: BLUEPRINT_TYPE needs to be imported here to register the blueprint


class PerTurnEvalResultsCompiler(AbstractResultsCompiler):
    """
    Compile and save results of human+model chats.

    Results will be saved on the level of specific conversations, as well as aggregated
    up the level of each worker as a whole.
    """

    # TODO: deduplicate setup_args from ModelChatResultsCompiler
    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--worker-block-list',
            type=str,
            default='',
            help='Comma-separated list of all workers to block',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):
        # TODO: deduplicate init from ModelChatResultsCompiler

        super().__init__(opt)

        # Input args
        os.makedirs(self.output_folder, exist_ok=True)
        self.worker_block_list = opt['worker_block_list'].split(',')

        # Save paths
        self.worker_results_path = os.path.join(
            self.output_folder, 'worker_results.csv'
        )
        self.unacceptable_worker_ids_path = os.path.join(
            self.output_folder, 'unacceptable_worker_ids.txt'
        )
        self.win_rate_by_date_path = os.path.join(
            self.output_folder, 'win_rates_by_date.csv'
        )
        self.stat_mean_length_by_date_path = os.path.join(
            self.output_folder, 'stat_mean_length_by_date.csv'
        )
        self.completion_time_by_model_pair_path = os.path.join(
            self.output_folder, 'mean_completion_times.csv'
        )

        self.acceptability_checker = AcceptabilityChecker()

        # Set fields that should be empty strings if the relevant information is not
        # present
        blank_field_columns = [
            'human_text',
            'human_choice',
            'human_justification',
            'accepted_bot_text',
            'not_accepted_bot_text',
        ]
        self.blank_fields = {field: '' for field in blank_field_columns}

        # Results attributes
        self.stat_counts = {}
        self.mean_completion_time = None
        # Useful for subclasses, to compare with other eval techniques

    def get_results_path_base(self) -> str:
        return os.path.join(self.output_folder, 'results')

    def compile_results(self) -> pd.DataFrame:

        # Load task data
        logging.info('Retrieving task data from Mephisto.')
        task_units_data = self.get_task_data()
        logging.info(f'Data for {len(task_units_data)} units loaded successfully.')

        # Read in each file
        num_convos_with_no_save_data = 0
        num_wrong_status_convos = 0
        num_complete_convos = 0
        worker_stats = {}
        worker_conversation_counts = {}
        total_utterances = 0

        unacceptable_task_units = []
        unacceptable_worker_ids = []
        conversation_idx = 0
        conversation_dfs = []

        for task_unit in task_units_data:

            worker_id = task_unit['worker_id']
            assignment_id = task_unit['assignment_id']

            # Determining whether the task unit should be skipped
            # Extract out custom data
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

            # Check if the Turker is on the list of blocked Turkers
            if worker_id in self.worker_block_list:
                logging.info(
                    f'Found a HIT with the worker {worker_id}, on the blocklist. '
                    f'Skipping.'
                )
                continue

            if worker_id in worker_conversation_counts:
                conversations_so_far = worker_conversation_counts[worker_id]
            else:
                conversations_so_far = 0
            worker_conversation_counts[worker_id] = conversations_so_far + 1

            data = task_unit['data']['save_data']['custom_data']

            # Extract out information about this conversation

            model_1_nickname = data['task_description']['model_1_nickname']
            model_2_nickname = data['task_description']['model_2_nickname']

            # Since we have two models, we use the format model_1_name:model_2_name
            model_pair_nickname = f"{model_1_nickname}:{model_2_nickname}"
            if model_pair_nickname not in self.stat_counts:
                self.stat_counts[model_pair_nickname] = {
                    'per_turn': defaultdict(
                        lambda: {model_1_nickname: 0, model_2_nickname: 0}
                    )
                }

            # Extract non-message info
            mturk_worker_id_match = re.fullmatch(
                r'--NOT-MTURK-AGENT-(.*)', data['workers'][0]
            )
            # TODO: figure out why --NOT-MTURK-AGENT appears at the beginning of this
            #  field, and remove it; then, remove this re.fullmatch() call
            if mturk_worker_id_match is not None:
                mturk_worker_id = mturk_worker_id_match.group(1)
            else:
                mturk_worker_id = None
            task_start = datetime.utcfromtimestamp(task_unit['task_start'])
            task_end = datetime.utcfromtimestamp(task_unit['task_end'])
            single_convo_info_dict = {
                'worker': worker_id,
                'mturk_worker_id': mturk_worker_id,
                'model_pair_nickname': model_pair_nickname,
                'bad_workers': ','.join(data['bad_workers']),
                'hit_id': data['hit_ids'][0],
                'assignment_id': assignment_id,
                'context_dataset': data['context_dataset'],
                'additional_context': data['additional_context'],
                'date': task_start.strftime('%Y-%m-%d'),
                'task_start': task_start,
                'task_end': task_end,
                'completion_time': (task_end - task_start).total_seconds(),
            }
            # TODO: 'task_start' and 'task_end' assume that the original datetime floats
            #  are stored in UTC. Check this!

            # Check that the conversation consists of pairs of comments between
            # agents 0 and 1, with 0 speaking first
            assert all(
                [
                    utterance_data['agent_idx'] == utterance_idx % 2
                    for utterance_idx, utterance_data in enumerate(data['dialog'])
                ]
            )
            messages_0 = [utt for utt in data['dialog'] if utt['agent_idx'] == 0]
            messages_1 = [utt for utt in data['dialog'] if utt['agent_idx'] == 1]
            assert len(messages_0) + len(messages_1) == len(data['dialog'])

            # Determine whether the HIT contains unacceptable messages.
            # (We do this for every HIT, even if acceptability violation info
            # was already saved, because the violation criteria may have
            # changed since the HIT was collected.)
            utterances_0 = [m['text'] for m in messages_0]
            assert utterances_0[0] == 'Hi!', (
                'This script assumes that the first human message is "Hi!", which is '
                'set by default and cannot be changed by the crowdsourcing worker.'
            )
            acceptability_violations = self.acceptability_checker.check_messages(
                messages=utterances_0[1:],  # Don't use the initial "Hi!"
                is_worker_0=True,
                violation_types=self.acceptability_checker.ALL_VIOLATION_TYPES,
            )
            if acceptability_violations != '':
                logging.info(
                    f'Conversation fails acceptability checks with a violation of '
                    f'"{acceptability_violations}", given the following utterances: '
                    f'{utterances_0[1:]}. Skipping.'
                )
                unacceptable_task_units.append(task_unit)
                assert (
                    mturk_worker_id is not None
                ), "MTurk worker ID cannot be determined for this unacceptable conversation!"
                unacceptable_worker_ids.append(mturk_worker_id)
                continue
            single_convo_info_dict[
                'acceptability_violations'
            ] = acceptability_violations

            # Identify information to put in each line of the output DataFrame
            info_for_each_turn = {
                'worker_id': single_convo_info_dict['worker'],
                'mturk_worker_id': single_convo_info_dict['mturk_worker_id'],
                'hit_id': single_convo_info_dict['hit_id'],
                'model_pair_nickname': model_pair_nickname,
                'conversation_idx': conversation_idx,
                'date': single_convo_info_dict['date'],
                'completion_time': single_convo_info_dict['completion_time'],
            }

            single_turn_dicts = []

            # Compile personas and previous utterances
            text_parts = []
            if data['personas'] is not None and len(data['personas']) > 0:
                assert len(data['personas']) == 2
                text_parts += [
                    'human persona: ' + ' '.join(data['personas'][0]),
                    'bot persona: ' + ' '.join(data['personas'][1]),
                ]
            if (
                data['additional_context'] is not None
                and len(data['additional_context']) > 0
            ):
                text_parts.append(data['additional_context'])
            single_turn_dicts.append(
                {
                    **info_for_each_turn,
                    'turn_idx': -1,
                    'agent_idx': -1,
                    'context': '\n'.join(text_parts),
                    **self.blank_fields,
                }
            )

            total_utterances += len([d for d in data["dialog"] if d["agent_idx"] == 1])
            if len(data['dialog']) > 20:
                logging.info(
                    f'Got long dialogue of {len(data["dialog"])} utterances, hit id: '
                    f'{single_convo_info_dict["hit_id"]}, model_pair_nickname: '
                    f'{model_pair_nickname}.'
                )

            # # Loop over conversation turns

            for utterance_idx, utt in enumerate(data['dialog']):

                this_turn_dict = {
                    **info_for_each_turn,
                    'turn_idx': utterance_idx,
                    'agent_idx': utt['agent_idx'],
                    **self.blank_fields,
                }

                if utt['agent_idx'] == 1:

                    # This is a turn in which the bots have responded

                    human_turn_idx = int(utterance_idx / 2) + 1
                    # Turns are 1-indexed

                    # TODO: maybe clean up some of this logic
                    this_turn_dict['human_choice'] = utt['human_choice']
                    this_turn_dict['human_justification'] = utt['human_justification']
                    this_turn_dict['accepted_bot_text'] = (
                        utt['accepted_bot_data']['text']
                        .replace('\n', '__newline__')
                        .replace('\r', '__CR__')
                    )
                    this_turn_dict['not_accepted_bot_text'] = (
                        utt['not_accepted_bot_data']['text']
                        .replace('\n', '__newline__')
                        .replace('\r', '__CR__')
                    )

                    if 'total' not in self.stat_counts[model_pair_nickname]:
                        self.stat_counts[model_pair_nickname]['total'] = 0
                    if this_turn_dict['agent_idx'] == 1:
                        self.stat_counts[model_pair_nickname]['total'] += 1

                    # Calculating overall human choice statistics
                    if model_1_nickname not in self.stat_counts[model_pair_nickname]:
                        self.stat_counts[model_pair_nickname][model_1_nickname] = 0
                    if model_2_nickname not in self.stat_counts[model_pair_nickname]:
                        self.stat_counts[model_pair_nickname][model_2_nickname] = 0

                    # Calculating per-turn human choice statistics
                    if utt['human_choice'] == model_1_nickname:
                        self.stat_counts[model_pair_nickname][model_1_nickname] += 1
                        self.stat_counts[model_pair_nickname]['per_turn'][
                            human_turn_idx
                        ][model_1_nickname] += 1
                    elif utt['human_choice'] == model_2_nickname:
                        self.stat_counts[model_pair_nickname][model_2_nickname] += 1
                        self.stat_counts[model_pair_nickname]['per_turn'][
                            human_turn_idx
                        ][model_2_nickname] += 1
                    else:
                        raise Exception(
                            'Something wrong has occurred: human choice is not equal '
                            'to either of the two models!'
                        )

                else:

                    # This is a turn in which the human has responded

                    this_turn_dict['human_text'] = utt['text']

                    # Counting some aspects of the human's utterances
                    if (
                        'human_utterance_count'
                        not in self.stat_counts[model_pair_nickname]
                    ):
                        self.stat_counts[model_pair_nickname][
                            'human_utterance_count'
                        ] = 0
                    self.stat_counts[model_pair_nickname]['human_utterance_count'] += 1

                    if 'human_word_count' not in self.stat_counts[model_pair_nickname]:
                        self.stat_counts[model_pair_nickname]['human_word_count'] = 0
                    self.stat_counts[model_pair_nickname]['human_word_count'] += len(
                        this_turn_dict['human_text'].strip().split(' ')
                    )

                    if (
                        'human_question_count'
                        not in self.stat_counts[model_pair_nickname]
                    ):
                        self.stat_counts[model_pair_nickname][
                            'human_question_count'
                        ] = 0
                    self.stat_counts[model_pair_nickname][
                        'human_question_count'
                    ] += this_turn_dict['human_text'].count('?')

                single_turn_dicts.append(this_turn_dict)

            # Finish up collecting per-conversation stats

            if single_convo_info_dict['worker'] not in worker_stats:
                worker_stats[single_convo_info_dict['worker']] = {'conversations': 0}
            worker_stats[single_convo_info_dict['worker']]['conversations'] += 1

            # Logic for calculating percent of conversations that are clean
            if 'acceptable_convos' not in self.stat_counts[model_pair_nickname]:
                self.stat_counts[model_pair_nickname]['acceptable_convos'] = 0
            self.stat_counts[model_pair_nickname]['acceptable_convos'] += 1

            # Adding the full conversation to the list of conversations
            single_convo_df = pd.DataFrame(single_turn_dicts)
            conversation_dfs.append(single_convo_df)
            conversation_idx += 1

        # Print results
        # TODO: all of this would be cleaner if saved as CSVs, so we don't have to
        #  re-run to get the results

        logging.info(
            f'{num_convos_with_no_save_data:d} conversations found with no save data.'
        )
        logging.info(
            f'{num_wrong_status_convos:d} conversations found with the wrong status.'
        )
        logging.info(f'{num_complete_convos:d} complete conversations found:')
        logging.info(f'\t{len(unacceptable_task_units):d} unacceptable conversations.')
        logging.info(f'\t{len(conversation_dfs):d} acceptable conversations.')
        for model_pair_nickname, model_stats_dict in self.stat_counts.items():
            logging.info(f'---{model_pair_nickname}---')
            model_1_nickname = model_pair_nickname.split(":")[0]
            model_2_nickname = model_pair_nickname.split(":")[1]
            for p, v in model_stats_dict.items():
                if p == 'per_turn':
                    for human_turn_idx in model_stats_dict['per_turn']:
                        per_turn_model_1 = model_stats_dict['per_turn'][human_turn_idx][
                            model_1_nickname
                        ]
                        per_turn_model_2 = model_stats_dict['per_turn'][human_turn_idx][
                            model_2_nickname
                        ]
                        per_turn_model_total = per_turn_model_1 + per_turn_model_2
                        logging.info(
                            f"Turn {human_turn_idx}, {model_1_nickname}: {per_turn_model_1} "
                            f"({per_turn_model_1/per_turn_model_total:.2%})"
                            f", {model_2_nickname}: {per_turn_model_2} "
                            f"({per_turn_model_2/per_turn_model_total:.2%})"
                        )
                    continue
                if p == 'human_word_count' or p == 'human_question_count':
                    logging.info(
                        f'{p}: {v} ({v/model_stats_dict["human_utterance_count"]:.3})'
                    )
                elif p == 'human_utterance_count':
                    logging.info(f'{p}: {v}')
                elif p == 'acceptable_convos':
                    logging.info(f'{p}: {v}')
                else:
                    logging.info(f'{p}: {v} ({v/model_stats_dict["total"]:.2%})')

        logging.info('Printing worker IDs not already in block list to add...')
        for b in unacceptable_task_units:
            worker_id = b['worker_id']
            if worker_id not in self.worker_block_list:
                logging.info(f"""'{worker_id}',""")
        logging.info('Done printing bad workers.')

        logging.info(f'\nWorker conversation counts: {worker_conversation_counts}')

        # Compile full results

        all_conversations_df = pd.DataFrame()
        for single_convo_df in conversation_dfs:
            all_conversations_df = all_conversations_df.append(single_convo_df)
        for field in self.blank_fields.keys():
            assert all_conversations_df[field].isna().sum() == 0, (
                f'Some rows of the "{field}" column have NaNs in them, making them '
                f'hard to calculate statistics on!'
            )

        # Save analysis files

        logging.info(
            f'Saving worker statistical results to {self.worker_results_path}.'
        )
        worker_columns = ['worker_id', 'conversations']
        worker_df = pd.DataFrame([], columns=worker_columns)
        for worker_id, data in worker_stats.items():
            stat = {'worker_id': worker_id, 'conversations': data['conversations']}
            worker_df = worker_df.append(stat, ignore_index=True)
        worker_df.to_csv(self.worker_results_path, index=False)

        logging.info(
            f'Saving MTurk IDs of workers with unacceptable conversations to '
            f'{self.unacceptable_worker_ids_path}.'
        )
        with open(self.unacceptable_worker_ids_path, 'w') as f:
            for worker_id in unacceptable_worker_ids:
                f.write(worker_id + '\n')

        logging.info(f'Saving win rates cut by date to {self.win_rate_by_date_path}.')
        pivoted_win_rate_df = (
            all_conversations_df[lambda df: df['human_choice'].notna()]
            .assign(count=1)
            .groupby(['model_pair_nickname', 'date', 'human_choice'])
            .agg({'count': 'sum'})
            .reset_index()
            .pivot(
                index=['model_pair_nickname', 'date'],
                columns='human_choice',
                values='count',
            )
        )
        model_names = pivoted_win_rate_df.columns
        pivoted_win_rate_df.loc[:, 'total_count'] = pivoted_win_rate_df[
            model_names
        ].sum(axis=1)
        for model_name in model_names:
            pivoted_win_rate_df.loc[:, f'frac_{model_name}'] = (
                pivoted_win_rate_df[model_name] / pivoted_win_rate_df['total_count']
            )
        pivoted_win_rate_df.to_csv(self.win_rate_by_date_path)

        logging.info(
            f'Saving mean word count of different stats, cut by date, to '
            f'{self.stat_mean_length_by_date_path}.'
        )
        stats_to_calculate_mean_length_of = ['human_text', 'human_justification']
        assert (
            len(stats_to_calculate_mean_length_of) == 2
        ), 'This section of the code won\'t work with more than 2 stats!'
        stat_mean_length_dfs = []
        for stat in stats_to_calculate_mean_length_of:
            stat_mean_length_dfs.append(
                all_conversations_df[lambda df: df[stat] != '']
                .assign(word_count=lambda df: df[stat].str.split().str.len())
                .groupby(['model_pair_nickname', 'date'])['word_count']
                .mean()
                .to_frame(stat)
            )
        joined_stat_mean_length_df = stat_mean_length_dfs[0].join(
            stat_mean_length_dfs[1]
        )
        joined_stat_mean_length_df.to_csv(self.stat_mean_length_by_date_path)

        logging.info(
            f'Saving mean completion time stats to '
            f'{self.completion_time_by_model_pair_path}.'
        )
        completion_time_by_convo_df = all_conversations_df[
            ['model_pair_nickname', 'conversation_idx', 'completion_time']
        ].drop_duplicates()
        for model_pair_nickname in completion_time_by_convo_df[
            'model_pair_nickname'
        ].unique():
            assert (
                completion_time_by_convo_df[
                    lambda df: df['model_pair_nickname'] == model_pair_nickname
                ].index.size
                == self.stat_counts[model_pair_nickname]['acceptable_convos']
            ), (
                f"The count of convos for the model pair {model_pair_nickname} is "
                f"inconsistent!"
            )
        completion_time_by_model_pair_df = (
            completion_time_by_convo_df.groupby('model_pair_nickname')[
                'completion_time'
            ]
            .mean()
            .to_frame('mean_completion_time')
        )
        completion_time_by_model_pair_df.to_csv(self.completion_time_by_model_pair_path)

        return all_conversations_df


if __name__ == '__main__':
    parser_ = PerTurnEvalResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    PerTurnEvalResultsCompiler(vars(args_)).compile_and_save_results()
