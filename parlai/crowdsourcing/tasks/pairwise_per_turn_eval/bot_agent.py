#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.crowdsourcing.tasks.model_chat.bot_agent import TurkLikeAgent
from parlai.core.message import Message
from parlai.utils.strings import normalize_reply


class PerTurnEvalTurkLikeAgent(TurkLikeAgent):
    """
    Will act like a Turker but actually contains a bot agent.
    """

    def __init__(self, opt, model_name, model_agent, num_turns, semaphore=None):
        super().__init__(opt, model_name, model_agent, num_turns, semaphore)

    def act(self, timeout=None):
        """
        Same as model chat's bot_agent.py except the self_observe function is removed, a
        custom observe is instead written in worlds.py.

        This is so that the two bots can read each others' messages using observe so
        that the conversation history stays the same.
        """

        _ = timeout  # The model doesn't care about the timeout

        if self.semaphore:
            with self.semaphore:
                act_out = self.model_agent.batch_act([self.model_agent.observation])[0]
        else:
            act_out = self.model_agent.batch_act([self.model_agent.observation])[0]
        act_out = Message(act_out).json_safe_payload()

        if 'dict_lower' in self.opt and not self.opt['dict_lower']:
            # model is cased so we don't want to normalize the reply like below
            final_message_text = act_out['text']
        else:
            final_message_text = normalize_reply(act_out['text'])

        act_out['text'] = final_message_text
        assert ('episode_done' not in act_out) or (not act_out['episode_done'])
        self.turn_idx += 1
        return {**act_out, 'episode_done': False}
