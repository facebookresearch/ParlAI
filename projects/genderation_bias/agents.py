#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.agents.transformer.transformer import TransformerGeneratorAgent


class BiasAgentTrait(object):
    """
    Trait that appends a string representing a gender bias class to the context.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        grp = super().add_cmdline_args(parser, partial_opt=partial_opt)
        grp.add_argument('--bias-class', type=str, default='f0m0')
        return parser

    def get_temp_history(self, observation):
        _ = observation  # Unused
        return ' ' + self.opt['bias_class']


class BiasAgent(BiasAgentTrait, TransformerGeneratorAgent):
    """
    Example usage: `-m projects.genderation_bias.agents:BiasAgent`.
    """

    pass
