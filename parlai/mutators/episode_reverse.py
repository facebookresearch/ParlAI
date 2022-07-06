#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import List, Optional
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, EpisodeMutator
from parlai.core.params import ParlaiParser


@register_mutator("episode_reverse")
class EpisodeReverseMutator(EpisodeMutator):
    """
    Reverses all the turns in a conversation.

    Labels remain in the original ordering, but the order of text (prompts) is mixed up.
    Thus "one half" of the conversation is reordered.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser.add_argument(
            '--preserve-context',
            default=True,
            type='bool',
            help='If extra context is provided, keep it prepended to the first turn',
        )

    def __init__(self, opt: Opt):
        super().__init__(opt)
        self.rng = random.Random(42)

    def episode_mutation(self, episode: List[Message]) -> List[Message]:
        texts = []
        for turn in episode:
            texts.append(turn.pop('text'))
        if self.opt.get('preserve_context'):
            first_turn = texts[0].split('\n')
            context, text = first_turn[:-1], first_turn[-1]
            texts[0] = text
        else:
            context = []
        texts = list(reversed(texts))
        for i, turn in enumerate(episode):
            text = texts.pop(0)
            if i == 0 and self.opt.get('preserve_context') and context:
                text = '\n'.join(context + [text])
            turn['text'] = text
        return episode
