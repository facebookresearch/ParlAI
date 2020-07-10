#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent for style-controlled generation.
"""

from parlai.agents.style_gen.modules import StyleAgentMixin, StyleHistory
from parlai.agents.transformer.transformer import TransformerGeneratorAgent


class StyleGenAgent(StyleAgentMixin, TransformerGeneratorAgent):
    """
    General purpose generator with a style in the history.
    """

    @classmethod
    def history_class(cls):
        """
        Determine the history class in order to append the style.
        """
        return StyleHistory

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add command-line arguments specifically for this agent.
        """
        StyleAgentMixin.add_cmdline_args(argparser)
        TransformerGeneratorAgent.add_cmdline_args(argparser)
        return argparser
