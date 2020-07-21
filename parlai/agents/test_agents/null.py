#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
from parlai.core.torch_agent import TorchAgent, Output


class NullAgent(TorchAgent):
    """
    A agent without train/eval_step.

    Useful if you only want preprocessing.
    """

    def __init__(self, opt, shared=None):
        self.model = self.build_model()
        self.criterion = self.build_criterion()
        super().__init__(opt, shared)

    def build_model(self):
        return torch.nn.Module()

    def build_criterion(self):
        return torch.nn.NLLLoss()

    def train_step(self, batch):
        return Output()

    def eval_step(self, batch):
        return Output()
