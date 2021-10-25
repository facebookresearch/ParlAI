#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from parlai.core.opt import Opt
from parlai.core.message import Message
from parlai.core.mutators import register_mutator, MessageMutator


@register_mutator("remove_doc")
class RemoveDoc(MessageMutator):
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
        text = message.pop('text')
        text = text.replace('<doc>\n', '')
        message['text'] = text
        return message


@register_mutator("add_label_to_input")
class AddLabelToInput(MessageMutator):
    """
    Adds the dialogue sentence to the input.

    But only a single time.
    """

    def message_mutation(self, message: Message) -> Message:
        new_message = message.copy()
        if 'text' not in message or 'labels' not in message or not message['labels']:
            return message
        if 'dialogue_response' in new_message:
            # checked_sentence_as_label was applied before
            labels = new_message['dialogue_response']
        else:
            labels = new_message['labels']
        dialogue_response = labels[0]
        text = new_message.pop('text')

        text += f'\n__label__ {dialogue_response} __endlabel__'
        new_message['text'] = text

        return new_message
