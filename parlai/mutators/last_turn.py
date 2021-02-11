#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Iterator
from parlai.core.message import Message
from parlai.core.mutators import ManyEpisodeMutator, register_mutator


@register_mutator("last_turn")
class LastTurnMutator(ManyEpisodeMutator):
    def many_episode_mutation(self, episode: List[Message]) -> Iterator[List[Message]]:
        history = []
        for message in episode:
            yield [message]
