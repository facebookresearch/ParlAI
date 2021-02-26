#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Iterator, Optional
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.message import Message
from parlai.core.mutators import ManyEpisodeMutator, register_mutator


@register_mutator("last_turn")
class LastTurnMutator(ManyEpisodeMutator):
    """
    Keep only the most recent turn.

    This mutator obliterates the history of the conversation, keeping only the very last
    thing said. Every turn is still evaluated, but treated as a new episode.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            '--preserve-context',
            default=True,
            type='bool',
            help='If extra context is provided, keep it prepended to all turns.',
        )

    def many_episode_mutation(self, episode: List[Message]) -> Iterator[List[Message]]:
        context = []
        for i, message in enumerate(episode):
            if i == 0 and self.opt.get('preserve_context'):
                text = message.pop('text').split("\n")
                context = text[:-1]
                message['text'] = text[-1]
            message['text'] = "\n".join(context + [message.pop('text')])
            yield [message]
