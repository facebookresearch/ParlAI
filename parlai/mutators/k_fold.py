#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional

from parlai.core.message import Message
from parlai.core.mutators import ManyEpisodeMutator, register_mutator
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.data import DatatypeHelper
import hashlib
import parlai.utils.logging as logging


"""
This introduces a k-fold cross validation mutator, that withholds one fold from the train set and then uses this fold for validation purpose.
"""


@register_mutator("k_fold_withhold_on_train")
class KFoldWithholdOnTrainMutator(ManyEpisodeMutator):
    """
    Cross validation mutator, withhold a fold.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt)
        agent = parser.add_argument_group('KFoldWithholdOnTrainMutator args')
        agent.add_argument(
            '--k-fold',
            type=int,
            help='the number of k-fold',
        )
        agent.add_argument(
            '--held-fold',
            type=int,
            help='the current fold reserved from training',
        )
        return parser

    def __init__(self, opt: Opt):
        super().__init__(opt)
        self.k_fold = opt['k_fold']
        self.held_fold = opt['held_fold']
        assert self.held_fold >= 0, f'held_fold should be greater than 0.'
        assert (
            self.held_fold < self.k_fold
        ), f'held_fold should be smaller than {self.k_fold}.'
        logging.info(
            f'We are doing {self.k_fold}-fold with {self.held_fold} held from training.'
        )
        self.is_training = DatatypeHelper.is_training(opt['datatype'])
        self.fold_cnt = 0
        return

    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        # assuming single message episode
        assert (
            len(episode) == 1
        ), 'k_fold mutator only works with single message episode.'
        if not self.is_training:
            return [episode]
        message = episode[0]
        text = message['text']
        fold = hashlib.sha256(text.encode('utf8')).digest()[0] % self.k_fold
        if fold == self.held_fold:
            self.fold_cnt += 1
            return []
        return [episode]

    def __del__(self):
        logging.info(f'withhold fold has {self.fold_cnt}')


@register_mutator("k_fold_release_on_valid")
class KFoldReleaseOnValidMutator(KFoldWithholdOnTrainMutator):
    # only use fold k for validation
    def many_episode_mutation(self, episode: List[Message]) -> List[List[Message]]:
        # assuming single message episode
        assert (
            len(episode) == 1
        ), 'k_fold mutator only works with single message episode.'
        if self.is_training:
            return [episode]
        message = episode[0]
        text = message['text']
        fold = hashlib.sha256(text.encode('utf8')).digest()[0] % self.k_fold
        if fold != self.held_fold:
            return []
        self.fold_cnt += 1
        return [episode]

    def __del__(self):
        logging.info(f'withhold fold has {self.fold_cnt}')
