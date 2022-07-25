#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

from parlai.core.message import Message
from parlai.core.mutators import MessageMutator, register_mutator
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.data import DatatypeHelper
import hashlib
import parlai.utils.logging as logging

import copy


"""
This mutator introduces noise to the training data by either fliping the labels or random set the labels,
work with binary classification only.
"""


@register_mutator("flip_classification_label")
class FlipClassificationLabelMutator(MessageMutator):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('FlipLabelMutator args')
        agent.add_argument(
            '--noise-level',
            type=float,
            help='The probability of label flipping',
        )
        agent.add_argument(
            '--classification-label-to-mutate',
            default='__ok__,__notok__',
            type=str,
            help='The binary labels in the data for flipping / random assignment. seprated by ,',
        )
        return parser

    def __init__(self, opt: Opt):
        super().__init__(opt)
        if opt['noise_level'] > 1 or opt['noise_level'] < 0:
            raise Exception('Noise_level should be within 0 - 1.')
        if opt['noise_level'] == 0:
            logging.info('No noise, teacher should be the same as before.')
        if opt['noise_level'] == 1:
            logging.info('Labels are exact the opposite.')
        self.labels = opt['classification_label_to_flip'].split(',')
        self.noise_level = opt['noise_level']
        logging.info(f'Flipping with noise level {self.noise_level}')
        self.is_training = DatatypeHelper.is_training(opt['datatype'])
        self.flip_cnt = 0
        return

    def message_mutation(self, message: Message) -> Message:
        message = copy.deepcopy(message)
        if 'labels' in message:
            labels = message['labels']
            text = message['text']
            new_labels = []
            flip = hashlib.md5(text.encode('utf8')).digest()[0] < self.noise_level * 256
            if len(labels) != 1:
                raise ValueError(
                    f'{type(self).__name__} can only be used with one label!'
                )
            else:
                # deterministic flipping for each example
                if flip:
                    self.flip_cnt += 1
                    if labels[0] == self.labels[0]:
                        new_labels = [self.labels[1]]
                    elif labels[0] == self.labels[1]:
                        new_labels = [self.labels[0]]
                    else:
                        raise ValueError(
                            f'labels must be binary and the same as in {self.labels}'
                        )
                    assert len(new_labels) == 1
            if flip:
                message.force_set('labels', new_labels)
        return message

    def __del__(self):
        logging.info(f'Flipped {self.flip_cnt}')


@register_mutator("flip_classification_label_train_only")
class FlipClassificationLabelTrainOnlyMutator(FlipClassificationLabelMutator):
    def __init__(self, opt: Opt):
        super().__init__(opt)
        if not self.is_training:
            logging.info('No flipping in eval / test mode.')
        return

    def message_mutation(self, message: Message) -> Message:
        if not self.is_training:
            return message
        else:
            return super().message_mutation(message)


@register_mutator("flip_classification_label_valid_only")
class FlipClassificationLabelValidOnlyMutator(FlipClassificationLabelMutator):
    def __init__(self, opt: Opt):
        super().__init__(opt)
        if self.is_training:
            logging.info('No flipping in train mode.')
        return

    def message_mutation(self, message: Message) -> Message:
        if self.is_training:
            return message
        else:
            return super().message_mutation(message)


@register_mutator("random_classification_label")
class RandomClassificationLabelMutator(FlipClassificationLabelMutator):
    def message_mutation(self, message: Message) -> Message:
        message = copy.deepcopy(message)
        if 'labels' in message:
            labels = message['labels']
            text = message['text']
            new_labels = []
            flip = hashlib.md5(text.encode('utf8')).digest()[0] < self.noise_level * 256
            if len(labels) != 1:
                raise ValueError(
                    f'{type(self).__name__} can only be used with one label!'
                )
            else:
                # deterministic flipping for each example
                if flip:
                    self.flip_cnt += 1
                    if hashlib.md5(text.encode('utf8')).digest()[1] / 256 > 0.5:
                        new_labels = [self.labels[0]]
                    else:
                        new_labels = [self.labels[1]]

                    assert len(new_labels) == 1
            if flip:
                message.force_set('labels', new_labels)
        return message
