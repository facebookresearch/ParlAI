#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.agents.image_seq2seq.image_seq2seq import ImageSeq2seqAgent
from projects.genderation_bias.agents import BiasAgentTrait


class BiasAgent(BiasAgentTrait, ImageSeq2seqAgent):
    """
    Example usage: `-m projects.multimodal_blenderbot.agents:BiasAgent`.
    """

    pass
