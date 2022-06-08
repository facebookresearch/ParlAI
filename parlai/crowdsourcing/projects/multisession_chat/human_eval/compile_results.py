#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from typing import Any, Dict
import numpy as np
import pandas as pd

from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.tasks.model_chat.analysis.compile_results import (
    ModelChatResultsCompiler as BaseModelChatResultsCompiler,
)
from parlai.crowdsourcing.utils.analysis import AbstractTurnAnnotationResultsCompiler


class ModelChatResultsCompiler(BaseModelChatResultsCompiler):
    """
    Compile and save results of human+model chats.

    Results will be saved on the level of specific conversations, as well as aggregated
    up the level of each worker as a whole.
    """

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--hit-block-list',
            type=str,
            default='',
            help='Comma-separated list of all hits to block',
        )
        parser.add_argument(
            '--results-folders', type=str, help='Comma-separated list of result folders'
        )
        parser.add_argument(
            '--model-nickname', type=str, default='', help='name of the model'
        )
        parser.add_argument(
            '--completed-run-stats-path',
            type=str,
            default='',
            help='path of the task run stats file',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        AbstractTurnAnnotationResultsCompiler.__init__(self, opt)
        if 'results_folders' in opt:
            self.results_folders = opt['results_folders'].split(',')
        else:
            self.results_folders = None

        # Input args
        self.model_nickname = opt['model_nickname']
        assert len(self.results_folders) > 0
        for folder in self.results_folders:
            assert os.path.isdir(folder), f'{folder} is not a valid folder!'
        os.makedirs(self.output_folder, exist_ok=True)
        self.start_date = opt['start_date']
        self.max_convos_per_worker = opt['max_convos_per_worker']
        self.min_word_count = opt['min_word_count']
        self.hit_block_list = opt['hit_block_list'].split(',')
        self.worker_block_list = opt['worker_block_list'].split(',')

        # Setting up problem buckets
        if self.use_problem_buckets:
            self.regular_buckets = [
                bucket
                for bucket in self.problem_buckets
                if bucket not in ['other', 'none_all_good']
            ]
            # Remove the buckets that are special cases

        self.acceptability_checker = AcceptabilityChecker()
        self.completed_run_stats_path = opt['completed_run_stats_path']

    def compile_results(self) -> pd.DataFrame:
        # TODO modularize the shared components to dedup the code
        read_folders = []
        date_strings = []
        import ipdb

        ipdb.set_trace()
        for folder in self.results_folders:
            # Load paths
            # TODO load this data in using DataBrowser
            date_strings = sorted(
                [
                    obj
                    for obj in os.listdir(folder)
                    if os.path.isdir(os.path.join(folder, obj))
                    and re.fullmatch(r'\d\d\d\d_\d\d_\d\d', obj)
                ]
            )
            if self.start_date != '':
                date_strings = [
                    str_ for str_ in date_strings if str_ >= self.start_date
                ]
            folders = [os.path.join(folder, str_) for str_ in date_strings]
            read_folders.extend(folders)
        print(f'Date folders: ' + ', '.join(date_strings))

        # Read in each file
        num_incomplete_convos = 0
        num_complete_convos = 0
        complete_convos_per_model = {}
        bad_conversations = []
        worker_stats = {}
        worker_conversation_counts = {}

        conversation_idx = 0
        conversation_dfs = []
        stat_counts = {}
        for read_folder in read_folders:
            read_folder_name = os.path.split(read_folder)[-1]
            for file_name in sorted(os.listdir(read_folder)):
                if file_name in self.hit_block_list or 'sandbox' in file_name:
                    continue

                if 'incomplete' in file_name:
                    num_incomplete_convos += 1
                    continue
                else:
                    num_complete_convos += 1

                # Read in file
                with open(os.path.join(read_folder, file_name), 'rb') as f:
                    data = json.load(f)

                # Only include the first max_convos_per_worker conversations from a
                # worker to avoid biasing
                worker_id = data['workers'][0]
                worker_id = worker_id.split('-')[-1]
                assignment_id = data['assignment_ids'][0]
                if worker_id in worker_conversation_counts:
                    conversations_so_far = worker_conversation_counts[worker_id]
                else:
                    conversations_so_far = 0
                worker_conversation_counts[worker_id] = conversations_so_far + 1
                if (
                    self.max_convos_per_worker != -1
                    and conversations_so_far >= self.max_convos_per_worker
                ):
                    print(
                        f'Had {conversations_so_far} conversation(s) already from this worker {worker_id}. Skipping {assignment_id}.'
                    )
                    continue

                # Check if need to block the turker
                word_counts = [
                    len(d['text'].split(' '))
                    for d in data['dialog']
                    if d['agent_idx'] == 0
                ]
                utterances = [d['text'] for d in data['dialog'] if d['agent_idx'] == 0]
                if np.average(word_counts) < self.min_word_count:
                    bad_conversations.append(data)
                    print(
                        f'Bad complete conversation, words from human: {utterances}. Skipping.'
                    )
                    continue

                if not all(
                    bucket in data['dialog'][0]['problem_data']
                    for bucket in self.problem_buckets
                ):
                    raise ValueError('Bucket(s) are missing from the problem data!')

                model_nickname = data['model_name']
                assert self.model_nickname == model_nickname
                initial_data_id = data['context_info']['observation_for_bot'][
                    'initial_data_id'
                ]
                if model_nickname not in stat_counts:
                    stat_counts[model_nickname] = {}
                if model_nickname in complete_convos_per_model:
                    complete_convos_per_model[model_nickname].append(initial_data_id)
                else:
                    complete_convos_per_model[model_nickname] = [initial_data_id]

                # Extract non-message info
                info_dict = {
                    'read_folder_name': read_folder_name,
                    'file_name': file_name,
                    'worker': worker_id,
                    'model_nickname': model_nickname,
                    'bad_workers': ','.join(data['bad_workers']),
                    'hit_id': data['hit_ids'][0],
                    'assignment_id': assignment_id,
                    'is_incomplete': 'incomplete' in file_name,
                    'context_info': data['context_info'],
                    'bot_persona_strings': data['bot_persona_strings'],
                    'human_persona_strings': data['human_persona_strings'],
                    'initial_task_data': data['initial_task_data'],
                    'initial_data_id': initial_data_id,
                }

                # Check that the conversation consists of pairs of comments between
                # agents 0 and 1, with 1(bot) speaking first
                assert all(
                    [
                        utterance_data['agent_idx'] == (utterance_idx + 1) % 2
                        for utterance_idx, utterance_data in enumerate(data['dialog'])
                    ]
                )

                # Determine whether the HIT contains unacceptable messages.
                # (We do this for every HIT, even if acceptability violation info
                # was already saved, because the violation criteria may have
                # changed since the HIT was collected.)
                messages_0 = [utt for utt in data['dialog'] if utt['agent_idx'] == 0]
                messages_1 = [utt for utt in data['dialog'] if utt['agent_idx'] == 1]
                assert len(messages_0) + len(messages_1) == len(data['dialog'])

                # Check the human utterances for safety
                utterances_0 = [m['text'] for m in messages_0]
                info_dict[
                    'acceptability_violations_0'
                ] = self.acceptability_checker.check_messages(
                    messages=utterances_0,
                    is_worker_0=True,
                    violation_types=self.acceptability_checker.ALL_VIOLATION_TYPES,
                )

                # Compile personas and previous utterances
                df = pd.DataFrame(
                    [],
                    columns=[
                        'folder',
                        'file_name' 'worker_id',
                        'hit_id',
                        'is_incomplete',
                        'context_info',
                        'initial_data_id',
                        'acceptability_violations_0',
                        'model_nickname',
                        'conversation_idx',
                        'turn_idx',
                        'agent_idx',
                        'text',
                    ]
                    + self.problem_buckets,
                )
                df = df.append(
                    {
                        'folder': info_dict['read_folder_name'],
                        'file_name': info_dict['file_name'],
                        'worker_id': info_dict['worker'],
                        'hit_id': info_dict['hit_id'],
                        'is_incomplete': info_dict['is_incomplete'],
                        'context_info': info_dict['context_info'],
                        'initial_data_id': info_dict['initial_task_data'],
                        'acceptability_violations_0': info_dict[
                            'acceptability_violations_0'
                        ],
                        'model_nickname': model_nickname,
                        'conversation_idx': conversation_idx,
                        'turn_idx': -1,
                        'agent_idx': 0,
                        'text': info_dict['context_info']['observation_for_bot'][
                            'text'
                        ],
                        **{bucket: '' for bucket in self.problem_buckets},
                    },
                    ignore_index=True,
                )

                for utterance_idx, utt in enumerate(data['dialog']):

                    d = {
                        'folder': info_dict['read_folder_name'],
                        'file_name': info_dict['file_name'],
                        'worker_id': info_dict['worker'],
                        'hit_id': info_dict['hit_id'],
                        'is_incomplete': info_dict['is_incomplete'],
                        'context_info': info_dict['context_info'],
                        'initial_data_id': info_dict['initial_task_data'],
                        'acceptability_violations_0': info_dict[
                            'acceptability_violations_0'
                        ],
                        'model_nickname': model_nickname,
                        'conversation_idx': conversation_idx,
                        'turn_idx': utterance_idx,
                        'agent_idx': utt['agent_idx'],
                        'text': utt['text'],
                        **{bucket: '' for bucket in self.problem_buckets},
                    }

                    if utt['agent_idx'] == 1:
                        if 'problem_data' not in utt:
                            for bucket in self.problem_buckets:
                                d[bucket] = 'MALFORMED'
                            print(
                                f'Warning got MALFORMED utterance problem data inside complete convo: {utt}. Skipping.'
                            )
                            continue
                        else:
                            for bucket in self.regular_buckets:
                                d[bucket] = utt['problem_data'][bucket]
                            d['final_rating'] = (
                                utt['final_rating'] if 'final_rating' in utt else None
                            )
                        for k in self.regular_buckets:
                            if k not in stat_counts[model_nickname]:
                                stat_counts[model_nickname][k] = 0
                            stat_counts[model_nickname][k] += d[k]

                        if 'total' not in stat_counts[model_nickname]:
                            stat_counts[model_nickname]['total'] = 0
                        if d['agent_idx'] == 1:
                            stat_counts[model_nickname]['total'] += 1
                        if d['final_rating'] is not None:
                            # Only one the last utterance (agent idx == 1)
                            if 'count_ratings' not in stat_counts[model_nickname]:
                                stat_counts[model_nickname]['count_ratings'] = 0
                            stat_counts[model_nickname]['count_ratings'] += 1
                            if 'ratings' not in stat_counts[model_nickname]:
                                stat_counts[model_nickname]['ratings'] = []
                            if 'pairwise_ratings' not in stat_counts[model_nickname]:
                                stat_counts[model_nickname]['pairwise_ratings'] = {}
                            stat_counts[model_nickname]['ratings'].append(
                                int(d['final_rating'])
                            )
                            stat_counts[model_nickname]['pairwise_ratings'][
                                info_dict['initial_data_id']
                            ] = int(d['final_rating'])

                        if 'bot_word_count' not in stat_counts[model_nickname]:
                            stat_counts[model_nickname]['bot_word_count'] = 0
                        stat_counts[model_nickname]['bot_word_count'] += len(
                            d['text'].strip().split(' ')
                        )
                    else:

                        # Counting some aspects of the human's utterances
                        if 'human_utterance_count' not in stat_counts[model_nickname]:
                            stat_counts[model_nickname]['human_utterance_count'] = 0
                        stat_counts[model_nickname]['human_utterance_count'] += 1

                        if 'human_word_count' not in stat_counts[model_nickname]:
                            stat_counts[model_nickname]['human_word_count'] = 0
                        stat_counts[model_nickname]['human_word_count'] += len(
                            d['text'].strip().split(' ')
                        )

                        if 'human_question_count' not in stat_counts[model_nickname]:
                            stat_counts[model_nickname]['human_question_count'] = 0
                        stat_counts[model_nickname]['human_question_count'] += d[
                            'text'
                        ].count('?')

                # Only want to count bot utterances but human ones, while included,
                # won't be False
                if info_dict['worker'] not in worker_stats:
                    worker_stats[info_dict['worker']] = {'conversations': 0}
                worker_stats[info_dict['worker']]['conversations'] += 1

                # Logic for calculating percent of conversations that are clean
                if 'count_convos' not in stat_counts[model_nickname]:
                    stat_counts[model_nickname]['count_convos'] = 0
                stat_counts[model_nickname]['count_convos'] += 1

                # Adding the full conversation to the list of conversations
                conversation_dfs.append(df)
                conversation_idx += 1

        for m, conversations_completed in complete_convos_per_model.items():
            print(
                f'Got {len(conversations_completed)} complete conversations for model: {m}'
            )
            print(f"{m} completed: {conversations_completed}")

        print(f'{num_complete_convos:d} complete conversation(s) collected.')
        print(f'{len(bad_conversations):d} bad conversation(s).')
        num_approved_convos = num_complete_convos - len(bad_conversations)
        print(f'{num_approved_convos:d} approved conversation(s).')
        print(f'({num_incomplete_convos:d} incomplete conversation(s) collected.)')
        for model_nickname, model_stats_dict in stat_counts.items():
            print(f'---{model_nickname}---')
            for p, v in model_stats_dict.items():
                if p == 'count_ratings' or p == 'pairwise_ratings':
                    continue
                if p == 'ratings':
                    print(
                        f'Average Engaging-ness Rating: {np.average(model_stats_dict["ratings"])} ({model_stats_dict["count_ratings"]} ratings)'
                    )
                    continue
                if p == 'human_word_count' or p == 'human_question_count':
                    print(
                        f'{p}: {v} ({v/model_stats_dict["human_utterance_count"]:.3})'
                    )
                elif p == 'bot_word_count':
                    print(f'{p}: {v} ({v/model_stats_dict["total"]:.3})')
                elif p == 'human_utterance_count':
                    print(f'{p}: {v}')
                elif p == 'count_convos':
                    print(f'{p}: {v}')
                else:
                    print(f'{p}: {v} ({v/model_stats_dict["total"]:.2%})')

        print('Printing worker IDs not already in block list to add...')
        for b in bad_conversations:
            worker_id = b['workers'][0]
            if worker_id not in self.worker_block_list:
                print(f"""'{worker_id}',""")
        print('Done printing bad workers.')

        worker_df = pd.DataFrame([], columns=['worker_id', 'conversations'])

        for worker_id, data in worker_stats.items():
            stat = {'worker_id': worker_id, 'conversations': data['conversations']}
            worker_df = worker_df.append(stat, ignore_index=True)

        with open(self.completed_run_stats_path, 'r') as f:
            completed_run_stats = json.load(f)
        assert completed_run_stats['bot_model_name'] == self.model_nickname
        completed_run_stats['context_done_statistics'][
            self.model_nickname
        ] = complete_convos_per_model[self.model_nickname]
        completed_run_stats['context_done_counts'] = len(
            complete_convos_per_model[self.model_nickname]
        )
        with open(self.completed_run_stats_path, 'w') as fw:
            json.dump(completed_run_stats, fw)
        print(f'Wrote override opt to: {self.completed_run_stats_path}')

        rating_path = os.path.join(self.output_folder, f'pairwise_ratings.json')
        with open(rating_path, 'w') as fw:
            json.dump(stat_counts[self.model_nickname]['pairwise_ratings'], fw)
        print(f'Wrote pairwise ratings to: {rating_path}')

        # Save full results
        all_conversations_df = pd.DataFrame()
        for df in conversation_dfs:
            all_conversations_df = all_conversations_df.append(df)

        return all_conversations_df


if __name__ == '__main__':
    """
    python parlai/crowdsourcing/projects/multisession_chat/human_eval/compile_results.py
    \

    --problem-buckets they,you,new,none,engaging \
    --model-nickname BST90M \
    --output-folder parlai/crowdsourcing/projects/multisession_chat/human_eval/results/BST90M \
    --results-folder parlai/crowdsourcing/projects/multisession_chat/human_eval/model_chat/BST90M \
    --completed-run-stats-path parlai/crowdsourcing/projects/multisession_chat/human_eval/task_config/completed_run_stats.json
    """
    parser_ = ModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    ModelChatResultsCompiler(vars(args_)).compile_and_save_results()
