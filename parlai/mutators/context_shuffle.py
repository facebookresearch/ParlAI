#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, MessageMutator


@register_mutator("context_shuffle")
class ContextShuffleMutator(MessageMutator):
    """
    Shuffles all the lines in the context.

    Context is defined to be any text prepended to the first turn of a conversation, for
    example, personas.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt)
        self.rng = random.Random(42)

    def message_mutation(self, message: Message) -> Message:
        texts = message.pop('text').split('\n')
        context, text = texts[:-1], texts[-1]
        self.rng.shuffle(context)
        output = context + [text]
        message['text'] = '\n'.join(output)
        return message
