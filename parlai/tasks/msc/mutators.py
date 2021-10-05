#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
from typing import List
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, ManyEpisodeMutator


@register_mutator("ltm_mutator")
class LongTermMemoryMutator(ManyEpisodeMutator):
    """
    Replaces the episode labels in messages with "personal_knwowledge".

    Episodes are flattened to ensure dialogue history is maintained appropriately.
    """

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        history = []
        for message in episode:
            if message['text'] == '__SILENCE__':
                continue
            history.append(message.pop('text'))
            message['text'] = '\n'.join(history)
            labels = message.pop('labels')
            message['labels'] = ["personal_knowledge"]
            yield [message]
            history.append(random.choice(labels))
