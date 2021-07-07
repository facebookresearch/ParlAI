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

TASK_DIR = '/private/home/jingxu23/ParlAI/parlai/crowdsourcing/tasks/model_chat'

MODEL_NICKNAMES = [
    'BST2.7B',
    'MSC2.7B_128',
    'MSC2.7B_1024',
    'RAG_SUMMSC',
    'FiD_SUMMSC',
    'FiDRAG_SUMMSC',
]

OVERRIDE_OPT_FILENAMES = {
    'MSC2.7B_128': 'override_opt_msc128.json',
    'MSC2.7B_1024': 'override_opt_msc1024.json',
    'BST2.7B': 'override_opt_bb.json',
    'RAG_SUMMSC': 'override_opt_ragsummsc.json',
    'FiD_SUMMSC': 'override_opt_fidsummsc.json',
    'FiDRAG_SUMMSC': 'override_opt_fidragsummsc.json',
    'RAG_RAWHISTORY_NDOC10': 'override_opt_ragrawhistoryndoc10.json',
    'KNOWLEDGE_BOT': 'override_opt_knowledgebot.json',
    'KNOWLEDGE_BOT_MEM': 'override_opt_knowledgebotmem.json',
    'KNOWLEDGE_SLUDGE_MEM': 'override_opt_knowledgesludgemem.json',
}


def get_override_opt_path(model_nickname, is_engaging):
    subfolder = 'engaging' if is_engaging else 'normal'
    return os.path.join(
        TASK_DIR, f'task_config/{subfolder}/{OVERRIDE_OPT_FILENAMES[model_nickname]}'
    )


def get_output_folder(model_nickname, is_engaging):
    subfolder = 'engaging' if is_engaging else 'normal'
    return os.path.join(TASK_DIR, f'results/{subfolder}/{model_nickname}')


def get_results_folder(model_nickname, is_engaging):
    subfolder = 'engaging' if is_engaging else 'normal'
    return [os.path.join(TASK_DIR, f'model_chat/{subfolder}/{model_nickname}')]


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
        parser.add_argument(
            '--model-nickname', type=str, default='', help='name of the model'
        )
        parser.add_argument(
            '--is-engaging', type=bool, default=False, help='whether if it is engaging'
        )
        return parser

    def __init__(self, opt: Dict[str, Any]):

        super().__init__(opt)

        # Input args
        self.model_nickname = opt['model_nickname']
        self.is_engaging = opt['is_engaging']
        self.results_folders = get_results_folder(self.model_nickname, self.is_engaging)
        assert len(self.results_folders) > 0
        for folder in self.results_folders:
            assert os.path.isdir(folder), f'{folder} is not a valid folder!'
        self.output_folder = get_output_folder(self.model_nickname, self.is_engaging)
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
        """
        python /private/home/jingxu23/ParlAI/parlai/crowdsourcing/tasks/model_chat/analy
        sis/compile_results.py \

            --problem-buckets they,you,new,none,contradiction \
            --model-nickname MSC2.7B_1024

        python /private/home/jingxu23/ParlAI/parlai/crowdsourcing/tasks/model_chat/analysis/compile_results.py \
            --problem-buckets they,you,new,none,engaging \
            --is-engaging True \
            --model-nickname BST2.7B
            FiD_SUMMSC
        """

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
        final_rating_stats = {}
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
                    bucket in data['dialog'][0]['problem_data']
                    for bucket in self.problem_buckets
                ):
                    raise ValueError('Bucket(s) are missing from the problem data!')
                # s = read_folder.split('/')[-2]
                # experimental_design = s[s.find('_') + 1 :]

                # model_nickname = experimental_design + '/' + data['workers'][1]
                model_nickname = data['model_name']
                assert self.model_nickname == model_nickname
                initial_data_id = data['context_info']['observation_for_bot'][
                    'initial_data_id'
                ]
                if model_nickname not in problem_counts:
                    problem_counts[model_nickname] = {}
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
                            if k not in problem_counts[model_nickname]:
                                problem_counts[model_nickname][k] = 0
                            problem_counts[model_nickname][k] += d[k]
                            if k == 'contradiction' and d.get(k, False):
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
                            if 'pairwise_ratings' not in problem_counts[model_nickname]:
                                problem_counts[model_nickname]['pairwise_ratings'] = {}
                            problem_counts[model_nickname]['ratings'].append(
                                int(d['final_rating'])
                            )
                            problem_counts[model_nickname]['pairwise_ratings'][
                                info_dict['initial_data_id']
                            ] = int(d['final_rating'])

                        if 'bot_word_count' not in problem_counts[model_nickname]:
                            problem_counts[model_nickname]['bot_word_count'] = 0
                        problem_counts[model_nickname]['bot_word_count'] += len(
                            d['text'].strip().split(' ')
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
                # is_problem = df['contradiction'].replace('', False)

                # Only want to count bot utterances but human ones, while included,
                # won't be False
                if info_dict['worker'] not in worker_stats:
                    worker_stats[info_dict['worker']] = {'conversations': 0}
                worker_stats[info_dict['worker']]['conversations'] += 1

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

        for m, conversations_completed in complete_convos_per_model.items():
            print(
                f'Got {len(conversations_completed)} complete conversations for model: {m}'
            )
            print(f"{m} completed: {conversations_completed}")

        print(f'{num_complete_convos:d} complete conversations collected.')
        print(f'{len(bad_conversations):d} bad conversations.')
        num_approved_convos = num_complete_convos - len(bad_conversations)
        print(f'{num_approved_convos:d} approved conversations.')
        print(f'({num_incomplete_convos:d} incomplete conversations collected.)')
        for model_nickname, model_problems_dict in problem_counts.items():
            print(f'---{model_nickname}---')
            for p, v in model_problems_dict.items():
                if p == 'count_ratings' or p == 'pairwise_ratings':
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
                elif p == 'bot_word_count':
                    print(f'{p}: {v} ({v/model_problems_dict["total"]:.3})')
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

        # print('Worker stats:')
        worker_df = pd.DataFrame([], columns=['worker_id', 'conversations'])

        for worker_id, data in worker_stats.items():
            # print(worker_id)
            stat = {'worker_id': worker_id, 'conversations': data['conversations']}
            worker_df = worker_df.append(stat, ignore_index=True)
        worker_df.to_csv(worker_results_file, index=False)
        # print(worker_df)
        print(f'Wrote worker statistical results to: {worker_results_file}')

        override_opt_path = get_override_opt_path(model_nickname, self.is_engaging)
        with open(override_opt_path) as f:
            override_opt = json.load(f)
        assert override_opt['bot_model_name'] == model_nickname
        override_opt['context_done_statistics'][
            model_nickname
        ] = complete_convos_per_model[model_nickname]
        override_opt['context_done_counts'] = len(
            complete_convos_per_model[model_nickname]
        )
        with open(override_opt_path, 'w') as fw:
            json.dump(override_opt, fw)
        print(f'Wrote override opt to: {override_opt_path}')

        rating_path = os.path.join(self.output_folder, f'pairwise_ratings.json')
        with open(rating_path, 'w') as fw:
            json.dump(problem_counts[model_nickname]['pairwise_ratings'], fw)
        print(f'Wrote pairwise ratings to: {rating_path}')

        # Save full results
        all_conversations_df = pd.DataFrame()
        for df in conversation_dfs:
            all_conversations_df = all_conversations_df.append(df)
        # print(f'\nWorker conversation counts: {worker_conversation_counts}')

        return all_conversations_df


if __name__ == '__main__':
    parser_ = ModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    ModelChatResultsCompiler(vars(args_)).compile_and_save_results()
