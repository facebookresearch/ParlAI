#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate the stacked models.
"""

from projects.k2r.stacked_agent.task.agents import (
    StackedKnowledgeDialogueAgent,
)
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript
from parlai.utils.strings import colorize
from parlai.core.worlds import _create_task_agents

import random
import json
from copy import deepcopy

from parlai.tasks.wizard_of_wikipedia.agents import (
    TOKEN_KNOWLEDGE,
    TOKEN_END_KNOWLEDGE,
)


def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(True, True, 'Print out examples for merged model.')
    parser.add_argument('-n', '--num-examples', type=int, default=5)
    parser.add_argument(
        '--interactive',
        type=bool,
        default=True,
        help='Interactively choose the knowledge response.',
    )
    parser.add_argument(
        '--save-file',
        type=str,
        default='',
        help='Save the responses as json file.',
    )
    parser.add_argument(
        '--random-order',
        type=bool,
        default=True,
        help='Go through the examples in random order.',
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=True,
        help='Print the examples.',
    )
    StackedKnowledgeDialogueAgent.add_cmdline_args(parser)
    return parser


def model_output(opt):
    # Init teacher and agent.
    teacher = _create_task_agents(opt)[0]
    stacked_agent = StackedKnowledgeDialogueAgent(opt)

    result = []

    entry_idx = 0
    episode_idx = 0
    seen_episodes = 0
    blocked_episode_prefixes = []
    while seen_episodes < opt['num_examples']:

        if opt['random_order']:
            # Hack to go through examples more randomly.
            for _ in range(random.randint(10, 30)):
                while not teacher.act()['episode_done']:
                    pass

        episode_done = False
        while not episode_done:
            # Interaction between teacher and agent.
            query = teacher.act()
            if blocked_episode_prefixes and any(
                [query['text'].startswith(pref) for pref in blocked_episode_prefixes]
            ):
                continue
            stacked_agent.observe(query)
            reply = stacked_agent.act()
            if episode_idx != teacher.episode_idx:
                entry_idx = 0
                seen_episodes += 1
                episode_idx = teacher.episode_idx
                if episode_idx == 1 or episode_idx % 5 == 0:
                    print(f'Episode {episode_idx}/{opt["num_examples"]}')
            episode_done = query['episode_done']

            # Get the gold labels for both the knowledge and dialogue response.
            label_key = 'eval_labels' if 'eval_labels' in query else 'labels'

            knowledge_response_target_kwords = [
                'knowledge_target',
                'knowledge_answer',
                'checked_sentence',
                '__selected-sentences__',
            ]
            target_knowledge_response = ''
            for kword in knowledge_response_target_kwords:
                if kword in query:
                    target_knowledge_response = query[kword]
                    break
            if 'SummaryQATeacher' in opt['task'] and not target_knowledge_response:
                target_knowledge_response = query['eval_labels'][0]
            if isinstance(target_knowledge_response, list):
                target_knowledge_response = ' '.join(target_knowledge_response)

            target_dialogue_response = reply.get(label_key, '')
            if not target_dialogue_response:
                target_dialogue_response = query.get('dialogue_response', '')
            if isinstance(target_dialogue_response, list):
                target_dialogue_response = ' '.join(target_dialogue_response)
            if (
                (
                    'lightqa_labeltype' in opt
                    and opt['lightqa_labeltype'] == 'dialogue_response'
                    and 'eval_labels' in query
                    and query['eval_labels']
                )
                or 'wizard_of_wikipedia' in opt['task']
                or 'light_dialog_wild' in opt['task']
            ):
                target_dialogue_response = query['eval_labels'][0]

            knowledge_response = reply.get('knowledge_response', '')
            dialogue_response = reply.get('text', '')

            result.append(
                dict(
                    episode_idx=episode_idx,
                    entry_idx=entry_idx,
                    context=query['text'],
                    knowledge_response=knowledge_response,
                    target_knowledge_response=target_knowledge_response,
                    dialogue_response=dialogue_response,
                    target_dialogue_response=target_dialogue_response,
                )
            )
            entry_idx += 1

            if opt['verbose']:
                # Print the history and predicted knowledge.
                print('\n', query['text'])
                print(
                    '  Knowledge Prediction: '
                    + colorize(knowledge_response, 'green')
                    + '  Gold: '
                    + colorize(target_knowledge_response, 'yellow')
                )
                if 'support_sentence' in reply:
                    print(
                        '  Support Sentence: '
                        + colorize(reply.get('support_sentence', ''), 'green')
                    )

                if opt['interactive']:
                    cont_choice = 'r'
                    while cont_choice == 'r':
                        # Let the user choose the conditioning knowledge.
                        user_input = input(
                            colorize(
                                'Type in knowledge to condition generation: ', 'red'
                            )
                        )

                        knowledge_infused_observation = deepcopy(
                            stacked_agent.observations['raw']
                        )
                        text = knowledge_infused_observation.pop('text')
                        text += (
                            f'\n{TOKEN_KNOWLEDGE} {user_input} {TOKEN_END_KNOWLEDGE}'
                        )
                        knowledge_infused_observation['text'] = text
                        dialogue_response_user_knowledge = stacked_agent.dialogue_reply(
                            agent=stacked_agent.dialogue_agent,
                            observation=knowledge_infused_observation,
                        )
                        dialogue_response_user_knowledge[
                            'knowledge_response'
                        ] = user_input
                        stacked_agent._filter_beams(
                            reply=dialogue_response_user_knowledge,
                            filter_for_knowledge=stacked_agent.opts['init'][
                                'beam_filter_for_knowledge_response'
                            ],
                            filter_questions=stacked_agent.opts['init'][
                                'beam_filter_questions'
                            ],
                            filter_self_references=stacked_agent.opts['init'][
                                'beam_filter_self_references'
                            ],
                        )
                        print(
                            '  Dialogue Prediction (user knowledge): '
                            + colorize(
                                dialogue_response_user_knowledge['text'], 'green'
                            )
                        )
                        cont_choice = input(
                            colorize('Continuation choice (c/d/n/r): ', 'red')
                        )
                        if cont_choice == 'd':
                            # Debug.
                            import pdb

                            pdb.set_trace()
                            cont_choice = 'c'
                        elif cont_choice == 'n':
                            # Skip the episode.
                            blocked_episode_prefixes = [
                                stacked_agent.observations['raw']['text']
                            ]
                            cont_choice = 'c'
                        elif cont_choice == 'r' or cont_choice == 'c':
                            # Continue or retry the episode.
                            pass
                        else:
                            print(
                                f"Can't parse continuation choice '{cont_choice}'. Continue."
                            )
                            cont_choice = 'c'

                # Print the remaining predictions.
                if (
                    'text_no_knowledge' in reply
                    and reply['text_no_knowledge'] != 'None'
                ):
                    print(
                        '  Dialogue Prediction (no knowledge): '
                        + colorize(reply.get('text_no_knowledge', ''), 'green')
                    )
                if 'text_rag_wiki' in reply and reply['text_rag_wiki'] != 'None':
                    print(
                        '  Dialogue Prediction (rag wiki): '
                        + colorize(reply.get('text_rag_wiki', ''), 'green')
                    )
                print(
                    '  Dialogue Prediction (predicted knowledge): '
                    + colorize(dialogue_response, 'green')
                )
                if (
                    'text_knowledge_sentence' in reply
                    and reply['text_knowledge_sentence'] != 'None'
                ):
                    print(
                        '  Dialogue Prediction (predicted knowledge sentence): '
                        + colorize(reply.get('text_knowledge_sentence', ''), 'green')
                    )

                if target_dialogue_response:
                    print(
                        '  Gold Dialogue Response: ',
                        colorize(target_dialogue_response, 'yellow'),
                    )

    if opt['save_file']:
        with open(opt['save_file'], 'w') as f:
            json.dump(result, f)
            print(f'Saved results to {opt["save_file"]}')


class StackedAgentOutput(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return model_output(self.opt)


if __name__ == '__main__':
    StackedAgentOutput.main()
