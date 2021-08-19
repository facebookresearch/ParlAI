#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import time
from typing import Any, Dict, List, Tuple

from parlai.core.worlds import validate
from parlai.crowdsourcing.tasks.model_chat.utils import Compatibility

from parlai.crowdsourcing.tasks.model_chat.worlds import (
    # ModelChatOnboardWorld,
    BaseModelChatWorld,
    get_bot_worker,
    get_world_params,
)

import parlai.utils.logging as logging


class ModelChatWorld(BaseModelChatWorld):
    """
    Version of BaseModelChatWorld for chatting without images.

    Has support for features that are currently not supported by the image-chat version
    of this task, like personas and BST-style seed utterances.
    """

    def __init__(self, opt, agent, bot, model_name, context_info):
        super().__init__(opt, agent=agent, bot=bot)

        self.context_info = context_info
        self.bot_persona_strings = self.context_info['your_persona_strings']
        self.human_persona_strings = self.context_info['their_persona_strings']
        self.context_for_bot_prompt = self.context_info['context_for_bot_prompt']
        self.time_num = self.context_info['time_num']
        self.time_unit = self.context_info['time_unit']
        self.task_data = {
            'personas': [
                " ".join(self.bot_persona_strings),
                " ".join(self.human_persona_strings),
            ],
            'agent_name': 'Speaker 1',
            'time_num': self.time_num,
            'time_unit': self.time_unit,
        }
        self.bot.agent_id = "THEY"
        self.model_name = model_name

    def _run_initial_turn(self) -> None:
        """
        Run the initial turn for both the human and the bot.

        Optionally show the bot its persona. If we are in BST conversation mode, show 2
        previous BST utterances to both the human and the bot; if we are in Meena-like
        conversation mode, show "Hi!" to the human and the bot and let the bot respond
        accordingly.
        """

        control_msg = {"episode_done": False}

        time.sleep(2)
        coordinator_first_msg = {
            'episode_done': False,
            'id': 'Coordinator',
            'text': 'Please chitchat with another worker for 6 turns as if you were catching up since last time you two spoke.',
            'fake_start': True,
            'agent_idx': 2,
            'task_data': self.task_data,
        }
        self.agent.observe(validate(coordinator_first_msg))
        time.sleep(2)
        human_first_msg = {
            'episode_done': False,
            'id': self.agent.id,
            'text': self.context_for_bot_prompt,
            'fake_start': True,
            'agent_idx': 0,
            'task_data': self.task_data,
        }
        for k, v in control_msg.items():
            human_first_msg[k] = v

        # self.dialog.append(human_first_msg)
        # self.agent.observe(validate(human_first_msg))
        print(human_first_msg)
        self.bot.observe(validate(human_first_msg))

        first_bot_act = self.bot.act()
        first_bot_act = Compatibility.maybe_fix_act(first_bot_act)
        first_bot_act['id'] = 'THEY'
        self.agent.observe(validate(first_bot_act))

        bot_utterance_data = {
            'agent_idx': 1,
            'text': first_bot_act['text'],
            'id': 'THEY',
        }
        self.dialog.append(bot_utterance_data)

    def get_final_chat_data(self) -> Dict[str, Any]:
        """
        Add non-image-chat-specific fields to the final chat data.
        """
        data = super().get_final_chat_data()
        context_data = {
            'model_name': self.model_name,
            'context_info': self.context_info,
            'bot_persona_strings': self.bot_persona_strings,
            'human_persona_strings': self.human_persona_strings,
            'initial_task_data': self.task_data,
        }
        data.update(context_data)
        return data

    def _prepare_acceptability_checking(self) -> Tuple[List[str], List[str]]:
        """
        Apply acceptability checking params specific to BST-style conversation.

        The BST mode starts the conversation with two previous utterances, so there
        should be no new greeting. Also, the first human response is one of the previous
        utterances, so it shouldn't get checked.
        """
        human_messages = [
            message['text'] for message in self.dialog if message['id'] == 'YOU'
        ]
        violation_types = ['min_words', 'all_caps', 'exact_match']
        if self.opt['conversation_start_mode'] == 'blended_skill_talk':
            violation_types.append('penalize_greetings')
            # human_messages = human_messages[1:]
        return human_messages, violation_types


def make_world(opt, agents, initialization_data):
    # TODO: merge this function in with the make_world in in parlai/crowdsourcing/tasks/model_chat/worlds.py
    # Extract important components from opt
    statistics_condition = opt['statistics_condition']
    context_generator = opt['context_generator']

    agents[0].agent_id = "YOU"

    # Decide on a bot to use
    run_statistics = opt['run_statistics']
    with statistics_condition:
        remaining_counts_needed = [
            (m, c - run_statistics[m]) for (m, c) in opt['conversations_needed'].items()
        ]
        remaining_counts_needed.sort(reverse=True, key=lambda x: x[1])
        model_name = remaining_counts_needed[0][0]
        print(f'Remaining conversation counts needed: {remaining_counts_needed}')
        print(f'Choosing the "{model_name}" model for the bot.')

    # Get context: personas, previous utterances, etc.
    if context_generator is not None:
        context_info = context_generator.get_context(model_name)
    else:
        context_info = None

    if context_info is None:
        logging.warning("SHOULD END THE TASK")
        return None

    bot_worker = get_bot_worker(opt=opt, model_name=model_name)

    return ModelChatWorld(
        opt,
        agent=agents[0],
        bot=bot_worker,
        model_name=model_name,
        context_info=context_info,
    )
