#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import pandas as pd

from parlai.crowdsourcing.tasks.model_chat.analysis.compile_results import (
    ModelChatResultsCompiler,
)


class MultiLIGHTModelChatResultsCompiler(ModelChatResultsCompiler):
    """
    Compile and save results of human+model chats, based on MultiLIGHT model chats.

    Results will be saved on the level of specific conversations, as well as aggregated
    up the level of each worker as a whole.
    """

    def compile_results(self) -> pd.DataFrame:
        task_units_data = self.get_task_data()
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

            # Only include the first max_convos_per_worker conversations from a
            # worker to avoid biasing
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

            persona_setting_message = task_unit['data']['messages'][2]
            personas = persona_setting_message['task_data']['personas']
            # The task always assigns the first persona to the human.
            human_speaker_character = personas[0]['name']

            saved_data = task_unit['data']['save_data']['custom_data']
            conv_data = saved_data['dialog']

            # Check if need to block the turker
            word_counts = [
                len(d['text'].split(' '))
                for d in conv_data
                if d['id'] == human_speaker_character
            ]
            human_utterances = [
                d['text'] for d in conv_data if d['id'] == human_speaker_character
            ]

            if np.average(word_counts) < self.min_word_count:
                bad_conversations.append(saved_data)
                print(
                    f'Bad complete conversation, words from human: {human_utterances}. Skipping.'
                )
                continue

            model_nickname = saved_data['task_description']['model_nickname']
            if model_nickname not in stat_counts:
                stat_counts[model_nickname] = {}

            if 'max_turn_rate' not in stat_counts[model_nickname]:
                stat_counts[model_nickname]['max_turn_rate'] = []

            if 'min_turn_rate' not in stat_counts[model_nickname]:
                stat_counts[model_nickname]['min_turn_rate'] = []

            if model_nickname in complete_convos_per_model:
                complete_convos_per_model[model_nickname] += 1
            else:
                complete_convos_per_model[model_nickname] = 1

            # Extract non-message info
            info_dict = {
                'worker': worker_id,
                'model_nickname': model_nickname,
                'bad_workers': ','.join(saved_data['bad_workers']),
                'hit_id': saved_data['hit_ids'][0],
                'assignment_id': assignment_id,
                'context_dataset': saved_data['context_dataset'],
                'additional_context': saved_data['additional_context'],
            }

            info_dict[
                'acceptability_violations_0'
            ] = self.acceptability_checker.check_messages(
                messages=human_utterances,
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
            for p in personas:
                text_parts.append(f'{p["name"]}: {p["persona"]}')

            new_row = pd.DataFrame(
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
                index=[0],
            )
            df = pd.concat(
                [df, new_row],
                ignore_index=True,
            )

            total_utterances += len(
                [d for d in saved_data["dialog"] if d['id'] == human_speaker_character]
            )
            if len(saved_data['dialog']) > 20:
                print(
                    f'Got long dialogue of {len(saved_data["dialog"])} utterances, hit id:'
                    f' {info_dict["hit_id"]}, model_nickname: {model_nickname}.'
                )

            speaker_count = dict()
            for p in personas:
                speaker_count[p['name'].lower()] = 0
            for utterance_idx, utt in enumerate(saved_data['dialog']):

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
                speaker_count[utt['id'].lower()] += 1

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
                            for bucket in self.regular_buckets + ['none']:
                                d[bucket] = utt['problem_data'][bucket]
                        for k in self.regular_buckets + ['none']:
                            if k not in stat_counts[model_nickname]:
                                stat_counts[model_nickname][k] = []
                            stat_counts[model_nickname][k].append(d[k])

                    if 'total' not in stat_counts[model_nickname]:
                        stat_counts[model_nickname]['total'] = 0
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

                df = pd.concat([df, pd.DataFrame(d, index=[0])], ignore_index=True)

            if info_dict['worker'] not in worker_stats:
                worker_stats[info_dict['worker']] = {'conversations': 0}
            worker_stats[info_dict['worker']]['conversations'] += 1

            # Logic for calculating percent of conversations that are clean
            if 'count_convos' not in stat_counts[model_nickname]:
                stat_counts[model_nickname]['count_convos'] = 0
            stat_counts[model_nickname]['count_convos'] += 1

            stat_counts[model_nickname]['max_turn_rate'].append(
                max(speaker_count.values())
            )
            stat_counts[model_nickname]['min_turn_rate'].append(
                min(speaker_count.values())
            )

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
                        f'Average Engaging-ness Rating: {np.average(model_stats_dict["ratings"])}'
                        f' ({model_stats_dict["count_ratings"]} ratings)'
                    )
                    print(
                        f'Engaging-ness Rating Variance: {np.std(model_stats_dict["ratings"])}'
                        f' ({model_stats_dict["count_ratings"]} ratings)'
                    )
                elif p == 'human_word_count' or p == 'human_question_count':
                    print(
                        f'{p}: {v} ({v/model_stats_dict["human_utterance_count"]:.3})'
                    )
                elif p == 'human_utterance_count':
                    print(f'{p}: {v}')
                elif p == 'count_convos':
                    print(f'{p}: {v}')
                elif p in ('min_turn_rate', 'max_turn_rate'):
                    print(f'{p}: {np.average(v)}')
                elif self.use_problem_buckets and p == 'convo_clean':
                    print(f'{p}: {v} ({v/model_stats_dict["count_convos"]:.2%})')
                else:
                    if p == 'total':
                        print(f'{p}: {v/model_stats_dict["total"]:.2%}')
                    else:
                        print(f'[DEBUG numpy] AVG {p}: {np.average(v):.2%}')
                        print(f'[DEBUG numpy] STD {p}: {np.std(v):.2%}')
                        print(f'{p}: {sum(v)/model_stats_dict["total"]:.2%}')

        print('Printing worker IDs not already in block list to add...')
        for b in bad_conversations:
            worker_id = b['workers'][0]
            if worker_id not in self.worker_block_list:
                print(f"""'{worker_id}',""")
        print('Done printing bad workers.')

        # Save full results
        all_conversations_df = pd.DataFrame()
        for df in conversation_dfs:
            all_conversations_df = pd.concat([all_conversations_df, df])
        print(f'\nWorker conversation counts: {worker_conversation_counts}')

        return all_conversations_df


if __name__ == '__main__':
    parser_ = MultiLIGHTModelChatResultsCompiler.setup_args()
    args_ = parser_.parse_args()
    MultiLIGHTModelChatResultsCompiler(vars(args_)).compile_and_save_results()
