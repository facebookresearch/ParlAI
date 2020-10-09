#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Implements extentions to NN code for transformers.

These include custom fine-tuning losses and the like that are training detail extentions applilcable to transformers rather than new model architectuures in themselves.
"""

import math
from typing import Dict, Tuple, Optional, Union

import numpy as np
import torch
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F

from parlai.core.torch_generator_agent import TorchGeneratorModel
from parlai.utils.misc import warn_once


class R3FNoiseTrait(object):
    @staticmethod
    def add_args(parser):
        parser.add_argument('--eps', type=float, default=1e-5, help='noise eps')
        parser.add_argument(
            '--r3f-lambda',
            type=float,
            default=1.0,
            help='lambda for combining logistic loss and noisy KL loss',
        )
        parser.add_argument(
            '--noise-type',
            type=str,
            default='uniform',
            choices=['normal', 'uniform'],
            help='type of noises for RXF methods',
        )

    def _get_symm_kl(self, noised_logits, input_logits):
        return (
            F.kl_div(
                F.log_softmax(noised_logits, dim=-1, dtype=torch.float32),
                F.softmax(input_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
            + F.kl_div(
                F.log_softmax(input_logits, dim=-1, dtype=torch.float32),
                F.softmax(noised_logits, dim=-1, dtype=torch.float32),
                None,
                None,
                "sum",
            )
        ) / noised_logits.size(0)


class R3FNoiseEncoderTrait(object):
    """
    Implements "Better Fine-Tuning by Reducing Representational Collapse"
 
    Paper: https://arxiv.org/abs/2008.03156
    """

    def __init__(self, *args, **kwargs):
        print("R3fEncoder Init")
        return


class R3FNoiseGeneratorAgentTrait(object):
    """
    Implements "Better Fine-Tuning by Reducing Representational Collapse"
 
    Paper: https://arxiv.org/abs/2008.03156
    """

    def __init__(self):
        pass


class R3FNoiseContext(object):
    def __init__(self):
        pass
