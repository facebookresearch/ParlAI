#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.message import Message


class BiasAgentTrait(object):
    """
    Abstract Trait.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        grp = super(BiasAgentTrait, cls).add_cmdline_args(argparser)
        grp.add_argument('-bias', '--bias-class', type=str, default='f0m0')

    def get_temp_history(self, observation):
        return ' ' + self.opt['bias_class']

    def observe2(self, observation):
        """
        Process incoming message in preparation for producing a response.

        This includes remembering the past history of the conversation.
        """
        observation = Message(observation)

        # Sanity check everything is in order
        self._validate_observe_invariants()

        if observation.get('episode_done'):
            self.__expecting_clear_history = True
        elif 'labels' in observation or 'eval_labels' in observation:
            # make sure we note that we're expecting a reply in the future
            self.__expecting_to_reply = True

        self.observation = observation
        # Update the history using the observation.
        # We may also consider adding a temporary string to the history
        # using the `get_temp_history()` function: this string will
        # persist until it is updated.
        self.history.update_history(
            observation, temp_history=self.get_temp_history(observation)
        )
        print("[ " + self.history.get_history_str() + " ] ")

        return self.vectorize(
            observation,
            self.history,
            text_truncate=self.text_truncate,
            label_truncate=self.label_truncate,
        )


class BiasAgent(BiasAgentTrait, TransformerGeneratorAgent):
    """
    Example usage:
    -m projects.genderation_bias.agents:BiasAgent
    """

    pass
