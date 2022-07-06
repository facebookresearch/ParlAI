#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Agent for style-controlled generation.
"""

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from projects.style_gen.modules import StyleAgentMixin


class StyleGenAgent(StyleAgentMixin, TransformerGeneratorAgent):
    """
    General purpose generator with a style in the history.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        """
        Add command-line arguments specifically for this agent.
        """
        StyleAgentMixin.add_cmdline_args(parser, partial_opt=partial_opt)
        TransformerGeneratorAgent.add_cmdline_args(parser, partial_opt=partial_opt)
        return parser
