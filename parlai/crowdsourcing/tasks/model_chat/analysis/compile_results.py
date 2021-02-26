#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.analysis import AbstractTurnAnnotationResultsCompiler


class ModelChatResultsCompiler(AbstractTurnAnnotationResultsCompiler):
    """
    Compile and save results of human+model chats.

    Results will be saved on the level of specific conversations, as well as aggregated
    up the level of each worker as a whole.
    """

    @classmethod
    def setup_args(cls):
        parser = super().setup_args()
        parser.add_argument(
            '--start-date',
            type=str,
            default='',
            help='The earliest date to analyze results from',
        )
        parser.add_argument(
            '--max-convos-per-worker',
            type=int,
            default=100,
            help='The most conversations to analyze from any one user. Set to -1 for no limit.',
        )
        parser.add_argument(
            '--min-word-count',
            type=int,
            default=4,
            help='The minimum acceptable mean number of words per human utterance',
        )
        parser.add_argument(
            '--hit-block-list',
            type=str,
            default='',
            help='Comma-separated list of all hits to block',
        )
        parser.add_argument(
            '--worker-block-list',
            type=str,
            default='',
            help='Comma-separated list of all workers to block',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        super().__init__(opt)

        # Input args
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
        self.regular_buckets = [
            bucket
            for bucket in self.problem_buckets
            if bucket not in ['other', 'none_all_good']
        ]
        # Remove the buckets that are special cases

        self.acceptability_checker = AcceptabilityChecker()

    def get_results_path_base(self) -> str:
        now = datetime.now()
        return os.path.join(
            self.output_folder, f'results_{now.strftime("%Y%m%d_%H%M%S")}'
        )

    def compile_results(self) -> pd.DataFrame:

        read_folders = []
        date_strings = []
        for folder in self.results_folders:
            # Load paths
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

        now = datetime.now()
        worker_results_file = os.path.join(
            self.output_folder, f'worker_results_{now.strftime("%Y%m%d_%H%M%S")}.csv'
        )
        # Read in each file
        num_incomplete_convos = 0
        num_complete_convos = 0
        complete_convos_per_model = {}
        bad_conversations = []
        problem_counts = {}
        worker_stats = {}
        worker_conversation_counts = {}
        total_utterances = 0

        conversation_idx = 0
        conversation_dfs = []
        for read_folder in read_folders:
            read_folder_name = os.path.split(read_folder)[-1]
            for file_name in sorted(os.listdir(read_folder)):
                if file_name in self.hit_block_list:
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
                        f'Had {conversations_so_far} conversations already from this worker {worker_id}. Skipping {assignment_id}.'
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
                    bucket in data['dialog'][1]['problem_data']
                    for bucket in self.problem_buckets
                ):
                    raise ValueError('Bucket(s) are missing from the problem data!')
                s = read_folder.split('/')[-2]
                experimental_design = s[s.find('_') + 1 :]

                model_nickname = experimental_design + '/' + data['workers'][1]
                if model_nickname not in problem_counts:
                    problem_counts[model_nickname] = {}
                if model_nickname in complete_convos_per_model:
                    complete_convos_per_model[model_nickname] += 1
                else:
                    complete_convos_per_model[model_nickname] = 1

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
                    'context_dataset': data['context_dataset'],
                    'additional_context': data['additional_context'],
                }

                # Check that the conversation consists of pairs of comments between
                # agents 0 and 1, with 0 speaking first
                assert all(
                    [
                        utterance_data['agent_idx'] == utterance_idx % 2
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
                        'worker_id',
                        'hit_id',
                        'model_nickname',
                        'conversation_idx',
                        'turn_idx',
                        'agent_idx',
                        'text',
                    ]
                    + self.problem_buckets,
                )
                if (
                    data['additional_context'] is not None
                    and len(data['additional_context']) > 0
                ):
                    additional_context = '\n' + data['additional_context']
                else:
                    additional_context = ''
                df = df.append(
                    {
                        'folder': info_dict['read_folder_name'],
                        'worker_id': info_dict['worker'],
                        'hit_id': info_dict['hit_id'],
                        'model_nickname': model_nickname,
                        'conversation_idx': conversation_idx,
                        'turn_idx': -1,
                        'agent_idx': 1,
                        'text': 'your persona: '
                        + data['personas'][1][0]
                        + '\nyour persona: '
                        + data['personas'][1][1]
                        + additional_context,
                        **{bucket: '' for bucket in self.problem_buckets},
                    },
                    ignore_index=True,
                )

                total_utterances += len(
                    [d for d in data["dialog"] if d["agent_idx"] == 1]
                )
                if len(data['dialog']) > 20:
                    print(
                        f'Got long dialogue of {len(data["dialog"])} utterances, hit id: {info_dict["hit_id"]}, model_nickname: {model_nickname}.'
                    )

                dialog_has_problems = False
                for utterance_idx, utt in enumerate(data['dialog']):
                    d = {
                        'folder': info_dict['read_folder_name'],
                        'worker_id': info_dict['worker'],
                        'hit_id': info_dict['hit_id'],
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
                            for bucket in self.regular_buckets + ['none_all_good']:
                                d[bucket] = utt['problem_data'][bucket]
                            d['final_rating'] = (
                                utt['final_rating'] if 'final_rating' in utt else None
                            )
                        for k in self.regular_buckets + ['none_all_good']:
                            if k not in problem_counts[model_nickname]:
                                problem_counts[model_nickname][k] = 0
                            problem_counts[model_nickname][k] += d[k]
                            if k != 'none_all_good' and d[k]:
                                dialog_has_problems = True

                        if 'total' not in problem_counts[model_nickname]:
                            problem_counts[model_nickname]['total'] = 0
                        if d['agent_idx'] == 1:
                            problem_counts[model_nickname]['total'] += 1
                        if d['final_rating'] is not None:
                            # Only one the last utterance (agent idx == 1)
                            if 'count_ratings' not in problem_counts[model_nickname]:
                                problem_counts[model_nickname]['count_ratings'] = 0
                            problem_counts[model_nickname]['count_ratings'] += 1
                            if 'ratings' not in problem_counts[model_nickname]:
                                problem_counts[model_nickname]['ratings'] = []
                            problem_counts[model_nickname]['ratings'].append(
                                int(d['final_rating'])
                            )
                    else:
                        # Counting some aspects of the human's utterances
                        if (
                            'human_utterance_count'
                            not in problem_counts[model_nickname]
                        ):
                            problem_counts[model_nickname]['human_utterance_count'] = 0
                        problem_counts[model_nickname]['human_utterance_count'] += 1

                        if 'human_word_count' not in problem_counts[model_nickname]:
                            problem_counts[model_nickname]['human_word_count'] = 0
                        problem_counts[model_nickname]['human_word_count'] += len(
                            d['text'].strip().split(' ')
                        )

                        if 'human_question_count' not in problem_counts[model_nickname]:
                            problem_counts[model_nickname]['human_question_count'] = 0
                        problem_counts[model_nickname]['human_question_count'] += d[
                            'text'
                        ].count('?')
                    df = df.append(d, ignore_index=True)

                # Count the number of problems the worker got
                is_problem = ~df['none_all_good'].replace('', True)

                # Only want to count bot utterances but human ones, while included,
                # won't be False
                count = is_problem.sum()
                if info_dict['worker'] not in worker_stats:
                    worker_stats[info_dict['worker']] = {
                        'conversations': 0,
                        'problems_found': 0,
                    }
                worker_stats[info_dict['worker']]['conversations'] += 1
                worker_stats[info_dict['worker']]['problems_found'] += count

                # Logic for calculating percent of conversations that are clean
                if 'count_convos' not in problem_counts[model_nickname]:
                    problem_counts[model_nickname]['count_convos'] = 0
                problem_counts[model_nickname]['count_convos'] += 1

                if not dialog_has_problems:
                    if 'convo_clean' not in problem_counts[model_nickname]:
                        problem_counts[model_nickname]['convo_clean'] = 0
                    problem_counts[model_nickname]['convo_clean'] += 1

                # Adding the full conversation to the list of conversations
                conversation_dfs.append(df)
                conversation_idx += 1

        for m, conversation_count in complete_convos_per_model.items():
            print(f'Got {conversation_count} complete conversations for model: {m}')

        print(f'{num_complete_convos:d} complete conversations collected.')
        print(f'{len(bad_conversations):d} bad conversations.')
        num_approved_convos = num_complete_convos - len(bad_conversations)
        print(f'{num_approved_convos:d} approved conversations.')
        print(f'({num_incomplete_convos:d} incomplete conversations collected.)')
        for model_nickname, model_problems_dict in problem_counts.items():
            print(f'---{model_nickname}---')
            for p, v in model_problems_dict.items():
                if p == 'count_ratings':
                    continue
                if p == 'ratings':
                    print(
                        f'Average Engaging-ness Rating: {np.average(model_problems_dict["ratings"])} ({model_problems_dict["count_ratings"]} ratings)'
                    )
                    continue
                if p == 'human_word_count' or p == 'human_question_count':
                    print(
                        f'{p}: {v} ({v/model_problems_dict["human_utterance_count"]:.3})'
                    )
                elif p == 'human_utterance_count':
                    print(f'{p}: {v}')
                elif p == 'count_convos':
                    print(f'{p}: {v}')
                elif p == 'convo_clean':
                    print(f'{p}: {v} ({v/model_problems_dict["count_convos"]:.2%})')
                else:
                    print(f'{p}: {v} ({v/model_problems_dict["total"]:.2%})')

        print('Printing worker IDs not already in block list to add...')
        for b in bad_conversations:
            worker_id = b['workers'][0]
            if worker_id not in self.worker_block_list:
                print(f"""'{worker_id}',""")
        print('Done printing bad workers.')

        print('Worker stats:')
        worker_df = pd.DataFrame(
            [],
            columns=[
                'worker_id',
                'conversations',
                'problems_found',
                'avg_problems_per_convo',
            ],
        )

        for worker_id, data in worker_stats.items():
            print(worker_id)
            avg_problems_per_convo = data['problems_found'] / data['conversations']
            stat = {
                'worker_id': worker_id,
                'conversations': data['conversations'],
                'problems_found': data['problems_found'],
                'avg_problems_per_convo': avg_problems_per_convo,
            }
            worker_df = worker_df.append(stat, ignore_index=True)
        worker_df = worker_df.sort_values('avg_problems_per_convo', ascending=0)
        worker_df.to_csv(worker_results_file, index=False)
        print(worker_df)
        print(f'Wrote worker statistical results to: {worker_results_file}')

        html_text = (
            '<html><body><table><tr><td>model</td>'
            + ''.join(f'<td>{bucket}</td>' for bucket in self.regular_buckets)
            + '<td>none_all_good</td><td>human_word_count</td><td>human_question_count</td><td>convo_clean</td><td>final_rating</td></tr>'
        )
        for model_nickname, model_problems_dict in problem_counts.items():
            html_text += f'<tr><td>{model_nickname}</td>'
            keys = self.regular_buckets + [
                'none_all_good',
                'human_word_count',
                'human_question_count',
            ]
            for k in keys:
                if k == 'human_word_count' or k == 'human_question_count':
                    html_text += f'<td>{model_problems_dict[k]} ({model_problems_dict[k]/model_problems_dict["human_utterance_count"]:.3} average)</td>'
                else:
                    html_text += f'<td>{model_problems_dict[k]} ({model_problems_dict[k]/model_problems_dict["total"]:.1%})</td>'
            html_text += f'<td>{model_problems_dict["convo_clean"]/model_problems_dict["count_convos"]:.1%} ({model_problems_dict["count_convos"]} convos)</td>'

            html_text += f'<td>{np.average(model_problems_dict["ratings"]):.2f} ({model_problems_dict["count_ratings"]} ratings)</td>'

            html_text += '</tr>'

        html_text += '</table>'
        print(html_text)

        # Save full results
        all_conversations_df = pd.DataFrame()
        for df in conversation_dfs:
            all_conversations_df = all_conversations_df.append(df)
        print(f'\nWorker conversation counts: {worker_conversation_counts}')

        return all_conversations_df


if __name__ == '__main__':
    parser_ = ModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    ModelChatResultsCompiler(vars(args_)).compile_and_save_results()
