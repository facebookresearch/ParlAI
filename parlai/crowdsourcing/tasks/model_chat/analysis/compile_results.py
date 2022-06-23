#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd

from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker
from parlai.crowdsourcing.utils.analysis import AbstractTurnAnnotationResultsCompiler
from parlai.crowdsourcing.tasks.model_chat.model_chat_blueprint import BLUEPRINT_TYPE

# importing BLUEPRINT_TYPE to force registration of the blueprint, not using this var itself
_ = BLUEPRINT_TYPE


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
            '--worker-block-list',
            type=str,
            default='',
            help='Comma-separated list of all workers to block',
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        super().__init__(opt)
        # Validate problem buckets
        if self.use_problem_buckets and 'none_all_good' not in self.problem_buckets:
            # The code relies on a catchall "none" category if the user selects no other
            # annotation bucket
            raise ValueError(
                'There must be a "none_all_good" category in self.problem_buckets!'
            )

        # Input args
        os.makedirs(self.output_folder, exist_ok=True)
        self.max_convos_per_worker = opt['max_convos_per_worker']
        self.min_word_count = opt['min_word_count']
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

    def get_results_path_base(self) -> str:
        now = datetime.now()
        return os.path.join(
            self.output_folder, f'results_{now.strftime("%Y%m%d_%H%M%S")}'
        )

    def compile_results(self) -> pd.DataFrame:
        task_units_data = self.get_task_data()
        now = datetime.now()
        worker_results_file = os.path.join(
            self.output_folder, f'worker_results_{now.strftime("%Y%m%d_%H%M%S")}.csv'
        )
        # Read in each file
        num_convos_with_no_save_data = 0
        num_wrong_status_convos = 0
        num_complete_convos = 0
        complete_convos_per_model = {}
        bad_conversations = []
        stat_counts = {}
        worker_stats = {}
        worker_conversation_counts = {}
        total_utterances = 0

        conversation_idx = 0
        conversation_dfs = []
        for task_unit in task_units_data:

            worker_id = task_unit['worker_id']
            assignment_id = task_unit['assignment_id']

            # Determining whether the task unit should be skipped
            # Extract out custom data
            if task_unit['data']['save_data'] is None:
                num_convos_with_no_save_data += 1
                continue
            elif task_unit['status'] not in ['completed', 'approved']:
                num_wrong_status_convos += 1
                continue
            else:
                num_complete_convos += 1

            # Read in file
            data = task_unit['data']['save_data']['custom_data']

            # Only include the first max_convos_per_worker conversations from a
            # worker to avoid biasing
            worker_id = task_unit['worker_id']
            assignment_id = task_unit['assignment_id']
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
                len(d['text'].split(' ')) for d in data['dialog'] if d['agent_idx'] == 0
            ]
            utterances = [d['text'] for d in data['dialog'] if d['agent_idx'] == 0]
            if np.average(word_counts) < self.min_word_count:
                bad_conversations.append(data)
                print(
                    f'Bad complete conversation, words from human: {utterances}. Skipping.'
                )
                continue

            if self.use_problem_buckets:
                if not all(
                    bucket in data['dialog'][1]['problem_data']
                    for bucket in self.problem_buckets
                ):
                    raise ValueError('Bucket(s) are missing from the problem data!')

            model_nickname = data['task_description']['model_nickname']
            if model_nickname not in stat_counts:
                stat_counts[model_nickname] = {}
            if model_nickname in complete_convos_per_model:
                complete_convos_per_model[model_nickname] += 1
            else:
                complete_convos_per_model[model_nickname] = 1

            # Extract non-message info
            info_dict = {
                'worker': worker_id,
                'model_nickname': model_nickname,
                'bad_workers': ','.join(data['bad_workers']),
                'hit_id': data['hit_ids'][0],
                'assignment_id': assignment_id,
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
            text_parts = []
            if data['personas'] is not None and len(data['personas']) > 0:
                text_parts += [
                    'your persona: ' + data['personas'][1][0],
                    'your persona: ' + data['personas'][1][1],
                ]
            if (
                data['additional_context'] is not None
                and len(data['additional_context']) > 0
            ):
                text_parts.append(data['additional_context'])
            df = df.append(
                {
                    'worker_id': info_dict['worker'],
                    'hit_id': info_dict['hit_id'],
                    'model_nickname': model_nickname,
                    'conversation_idx': conversation_idx,
                    'turn_idx': -1,
                    'agent_idx': 1,
                    'text': '\n'.join(text_parts),
                    **{bucket: '' for bucket in self.problem_buckets},
                },
                ignore_index=True,
            )

            total_utterances += len([d for d in data["dialog"] if d["agent_idx"] == 1])
            if len(data['dialog']) > 20:
                print(
                    f'Got long dialogue of {len(data["dialog"])} utterances, hit id: {info_dict["hit_id"]}, model_nickname: {model_nickname}.'
                )

            if self.use_problem_buckets:
                dialog_has_problems = False
            for utterance_idx, utt in enumerate(data['dialog']):

                d = {
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

                    d['final_rating'] = utt.get('final_rating')

                    if self.use_problem_buckets:
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
                        for k in self.regular_buckets + ['none_all_good']:
                            if k not in stat_counts[model_nickname]:
                                stat_counts[model_nickname][k] = 0
                            stat_counts[model_nickname][k] += d[k]
                            if k != 'none_all_good' and d[k]:
                                dialog_has_problems = True

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
                        stat_counts[model_nickname]['ratings'].append(
                            int(d['final_rating'])
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

                d = self._add_additional_per_turn_stats(d=d, utt=utt)

                df = df.append(d, ignore_index=True)

            if info_dict['worker'] not in worker_stats:
                worker_stats[info_dict['worker']] = {'conversations': 0}
                if self.use_problem_buckets:
                    worker_stats[info_dict['worker']]['problems_found'] = 0
            worker_stats[info_dict['worker']]['conversations'] += 1

            if self.use_problem_buckets:
                # Count the number of problems the worker got
                is_problem = ~df['none_all_good'].replace('', True)
                # Only want to count bot utterances but human ones, while included,
                # won't be False
                count = is_problem.sum()
                worker_stats[info_dict['worker']]['problems_found'] += count

            # Logic for calculating percent of conversations that are clean
            if 'count_convos' not in stat_counts[model_nickname]:
                stat_counts[model_nickname]['count_convos'] = 0
            stat_counts[model_nickname]['count_convos'] += 1

            if self.use_problem_buckets and not dialog_has_problems:
                if 'convo_clean' not in stat_counts[model_nickname]:
                    stat_counts[model_nickname]['convo_clean'] = 0
                stat_counts[model_nickname]['convo_clean'] += 1

            # Adding the full conversation to the list of conversations
            conversation_dfs.append(df)
            conversation_idx += 1

        for m, conversation_count in complete_convos_per_model.items():
            print(f'Got {conversation_count} complete conversation(s) for model: {m}')

        print(f'{num_complete_convos:d} complete conversation(s) collected.')
        print(f'{len(bad_conversations):d} bad conversation(s).')
        num_approved_convos = num_complete_convos - len(bad_conversations)
        print(f'{num_approved_convos:d} approved conversation(s).')
        print(f'({num_wrong_status_convos:d} wrong status conversation(s) collected.)')
        print(
            f'({num_convos_with_no_save_data:d} conversation(s) collected with no saved data.)'
        )
        for model_nickname, model_stats_dict in stat_counts.items():
            print(f'---{model_nickname}---')
            for p, v in model_stats_dict.items():
                if p == 'count_ratings':
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
                elif p == 'human_utterance_count':
                    print(f'{p}: {v}')
                elif p == 'count_convos':
                    print(f'{p}: {v}')
                elif self.use_problem_buckets and p == 'convo_clean':
                    print(f'{p}: {v} ({v/model_stats_dict["count_convos"]:.2%})')
                else:
                    print(f'{p}: {v} ({v/model_stats_dict["total"]:.2%})')

        print('Printing worker IDs not already in block list to add...')
        for b in bad_conversations:
            worker_id = b['workers'][0]
            if worker_id not in self.worker_block_list:
                print(f"""'{worker_id}',""")
        print('Done printing bad workers.')

        print('Worker stats:')
        worker_columns = ['worker_id', 'conversations']
        if self.use_problem_buckets:
            worker_columns += ['problems_found', 'avg_problems_per_convo']
        worker_df = pd.DataFrame([], columns=worker_columns)

        for worker_id, data in worker_stats.items():
            print(worker_id)

            stat = {'worker_id': worker_id, 'conversations': data['conversations']}
            if self.use_problem_buckets:
                avg_problems_per_convo = data['problems_found'] / data['conversations']
                stat.update(
                    {
                        'problems_found': data['problems_found'],
                        'avg_problems_per_convo': avg_problems_per_convo,
                    }
                )
            worker_df = worker_df.append(stat, ignore_index=True)
        if self.use_problem_buckets:
            worker_df = worker_df.sort_values('avg_problems_per_convo', ascending=0)
        worker_df.to_csv(worker_results_file, index=False)
        print(worker_df)
        print(f'Wrote worker statistical results to: {worker_results_file}')

        # Save full results
        all_conversations_df = pd.DataFrame()
        for df in conversation_dfs:
            all_conversations_df = all_conversations_df.append(df)
        print(f'\nWorker conversation counts: {worker_conversation_counts}')

        return all_conversations_df

    def _add_additional_per_turn_stats(self, d: dict, utt: dict) -> dict:
        """
        Add in additional statistics on the level of each conversation turn.

        Useful for subclasses.
        """
        _ = utt  # utt is ignored in this passthrough method
        return d


if __name__ == '__main__':
    parser_ = ModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    ModelChatResultsCompiler(vars(args_)).compile_and_save_results()
