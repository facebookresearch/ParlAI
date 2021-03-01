#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, MessageMutator


@register_mutator("word_shuffle")
class WordShuffleMutator(MessageMutator):
    """
    Shuffles all the words in the message.

    Only the text (prompt) is modified, not the labels. Utterances separated by newlines
    will not be shuffled across boundaries. You may wish to combine it with the flatten
    mutator to shuffle labels and texts.
    """

    def __init__(self, opt: Opt):
        super().__init__(opt)
        self.rng = random.Random(42)

    def message_mutation(self, message: Message) -> Message:
        texts = message.pop('text').split('\n')
        output_texts = []
        for text in texts:
            words = text.split(' ')
            self.rng.shuffle(words)
            output_texts.append(' '.join(words))
        message['text'] = '\n'.join(output_texts)
        return message
