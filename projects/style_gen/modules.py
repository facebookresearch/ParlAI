#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Models and helper classes for style-controlled generation.
"""

import random

from parlai.core.torch_agent import History


STYLE_SEP_TOKEN = ' STYLE '


class StyleHistoryMixin(History):
    """
    Methods for adding style to history.
    """

    def __init__(self, opt, **kwargs):
        super().__init__(opt, **kwargs)
        self.use_style_frac = opt['use_style_frac']
        self.style = None

    def reset(self):
        super().reset()
        self.style = None

    def update_history(self, obs, *args, **kwargs):
        super().update_history(obs, *args, **kwargs)
        use_style_rand = random.random()
        if use_style_rand < self.use_style_frac:
            # Use the style
            self.style = obs.get('personality')
            # This key name is dependent on Image-Chat and will change for other tasks.
            # If obs does not contain 'personality' (i.e. at the end of an epoch during
            # validation), there will be no style
            if self.style == '':
                self.style = None
        else:
            self.style = None

    def get_history_str(self):
        history_str = super().get_history_str()
        if history_str is not None and self.style is not None:
            history_str += STYLE_SEP_TOKEN + self.style

        return history_str

    def get_history_vec(self):
        history = super().get_history_vec()

        if history is not None and self.style is not None:
            style = STYLE_SEP_TOKEN + self.style
            style_tok = self.parse(style)
            history += style_tok

        return history


class StyleAgentMixin:
    """
    Methods for agents that return style from their histories.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.

        Does not add arguments from its superclass because it's a mixin.
        """
        agent = argparser.add_argument_group('StyleAgentMixin arguments')
        agent.add_argument(
            '--use-style-frac',
            type=float,
            default=1.0,
            help='What fraction of the time to use the style label',
        )
        return agent


class StyleHistory(StyleHistoryMixin, History):
    """
    Modify history to save the style.
    """
