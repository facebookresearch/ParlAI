#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Models and helper classes for style-controlled generation.
"""

import random
from typing import Optional

from parlai.core.message import Message
from parlai.core.opt import Opt


STYLE_SEP_TOKEN = ' STYLE '


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

    def __init__(self, opt: Opt, shared=None):
        super().__init__(opt, shared)
        self.use_style_frac = opt['use_style_frac']

    def get_temp_history(self, observation: Message) -> Optional[str]:
        """
        Conditionally return a style-token string to temporarily insert into history.
        """
        use_style_rand = random.random()
        if use_style_rand < self.use_style_frac:
            # Use the style
            style = observation.get('personality')
            # This key name is dependent on Image-Chat and will change for other tasks.
            # If obs does not contain 'personality' (i.e. at the end of an epoch during
            # validation), there will be no style
        else:
            style = ''
        if style is not None and style != '':
            return STYLE_SEP_TOKEN + style
