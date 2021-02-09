#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, EpisodeMutator


@register_mutator("episode_shuffle")
class EpisodeShuffleMutator(EpisodeMutator):
    """
    Shuffles all the turns in a conversation.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt)
        self.rng = random.Random(42)

    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        self.rng.shuffle(episode)
        return episode
