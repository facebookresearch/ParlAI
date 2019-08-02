#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test TorchAgent."""

from parlai.core.torch_agent import TorchAgent, Output
import torch
from parlai.core.agents import Agent


class MockDict(Agent):
    """Mock Dictionary Agent which just implements indexing and txt2vec."""

    null_token = '__null__'
    NULL_IDX = 0
    start_token = '__start__'
    BEG_IDX = 1001
    end_token = '__end__'
    END_IDX = 1002
    p1_token = '__p1__'
    P1_IDX = 2001
    p2_token = '__p2__'
    P2_IDX = 2002

    def __init__(self, opt, shared=None):
        """Initialize idx for incremental indexing."""
        self.idx = 0

    def __getitem__(self, key):
        """Return index of special token or return the token."""
        if key == self.null_token:
            return self.NULL_IDX
        elif key == self.start_token:
            return self.BEG_IDX
        elif key == self.end_token:
            return self.END_IDX
        elif key == self.p1_token:
            return self.P1_IDX
        elif key == self.p2_token:
            return self.P2_IDX
        else:
            self.idx += 1
            return self.idx

    def __setitem__(self, key, value):
        pass

    def __len__(self):
        return 0

    def add_cmdline_args(self, *args, **kwargs):
        """Add CLI args."""
        pass

    def txt2vec(self, txt):
        """Return index of special tokens or range from 1 for each token."""
        self.idx = 0
        return [self[tok] for tok in txt.split()]

    def save(self, path, sort=False):
        pass


class TorchAgent(TorchAgent):
    """Use MockDict instead of regular DictionaryAgent."""

    # def __init__(self, opt, shared=None):
    #     super().__init__(opt, shared)
    #     if not shared:
    #         self.init_model_file, self.is_finetune = self._get_init_model(opt, shared)
    #     else:
    #         self.init_model_file = shared['init_model_file']
    #         self.is_finetune = shared['is_finetune']
    #
    # def share(self):
    #     """Override to include init_model."""
    #     shared = super().share()
    #     shared['init_model_file'] = self.init_model_file
    #     shared['is_finetune'] = self.is_finetune

    @staticmethod
    def dictionary_class():
        """Replace normal dictionary class with mock one."""
        return MockDict

    def train_step(self, batch):
        """Return confirmation of training."""
        return Output(['Training {}!'.format(i) for i in range(len(batch.text_vec))])

    def eval_step(self, batch):
        """Return confirmation of evaluation."""
        return Output(['Evaluating {}!'.format(i) for i in range(len(batch.text_vec))])
