#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Teachers used in the Am I Me or You task.
"""
import re
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, EpisodeMutator
from typing import List

from projects.light_whoami.task.utils import (
    DEFAULT_DELIM,
    CONTEXT_KEYS,
    extract_characters,
)


@register_mutator('clean_context_mutator')
class CleanContextMutator(EpisodeMutator):
    """
    Context Mutator for LIGHT that removes random double-spaces from the context.
    """

    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        context_ep = episode[0]
        delimiter = self.opt.get('delimiter', DEFAULT_DELIM)
        context_str = context_ep['text'].split('\n')
        context = [s for s in context_str if any(t in s for t in CONTEXT_KEYS)]
        non_context = [s for s in context_str if not any(t in s for t in CONTEXT_KEYS)]
        context = '\n'.join([re.sub(r'[ ]+', ' ', s) for s in context])
        cleaned_str = delimiter.join([context] + non_context)
        context_ep.force_set('text', cleaned_str)
        episode[0] = context_ep
        return episode


@register_mutator('share_self_character')
class ShareSelfCharacterMutator(EpisodeMutator):
    """
    Mutator that provides self character in a special observation field.
    """

    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        characters = extract_characters(episode[0]['text'])
        for ep in episode:
            ep.force_set('self_character', characters['_self_name'])
        return episode


@register_mutator('left_to_right')
class LeftToRightMutator(EpisodeMutator):
    """
    Mutator that breaks down episodes into all partial sequences.
    """

    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        new_episode = []
        for ep in episode:
            label_words = ep.pop('labels')[0].split()
            for i in range(1, len(label_words) + 1):
                new_episode.append(
                    Message({**ep, 'labels': [' '.join(label_words[:i])]})
                )
        return new_episode
