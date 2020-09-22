#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
UnigramAgent always predicts the unigram distribution.

It is a full TorchGeneratorAgent model, so it can be used heavily in testing, while
being very quick to optimize.
"""

import torch
import torch.nn as nn
from parlai.core.torch_generator_agent import TorchGeneratorAgent, TorchGeneratorModel


class UnigramEncoder(nn.Module):
    def forward(self, x):
        return None


class UnigramDecoder(nn.Module):
    def forward(self, x, encoder_state, incr_state=None):
        return x.unsqueeze(-1), None


class UnigramModel(TorchGeneratorModel):
    def __init__(self, dictionary):
        super().__init__()
        self.encoder = UnigramEncoder()
        self.decoder = UnigramDecoder()
        self.v = len(dictionary)
        self.p = nn.Parameter(torch.zeros(self.v))

    def output(self, do):
        desired = list(do.shape)[:2] + [self.v]
        x = self.p.unsqueeze(0).unsqueeze(0)
        return x.expand(desired)

    def reorder_encoder_states(self, *args):
        return None

    def reorder_decoder_incremental_state(self, *args):
        return None


class UnigramAgent(TorchGeneratorAgent):
    def build_model(self):
        return UnigramModel(self.dict)
