#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List
from parlai.core.message import Message
from parlai.core.mutators import ManyEpisodeMutator, register_mutator


@register_mutator("flatten")
class FlattenMutator(ManyEpisodeMutator):
    """
    Flattens the entire conversation history.

    Simply concatenates all turns in the conversation with a newline. Frequently useful
    when composed with other mutators.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        history = []
        for message in episode:
            history.append(message.pop('text'))
            message['text'] = '\n'.join(history)
            yield [message]
            history.append(random.choice(message['labels']))
