# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.core.torch_agent import TorchAgent
from parlai.core.build_data import modelzoo_path
from parlai.core.dict import DictionaryAgent
# from .modules import GenImageCaption
from parlai.core.utils import round_sigfigs

import torch
import torch.nn as nn

import torchvision.transforms as transforms
from torch.nn.utils.rnn import pack_padded_sequence

import math
import os
import random


class VSEppAgent(TorchAgent):
    @staticmethod
    def add_cmdline_args(argparser):
        """Add command-line arguments specifically for this agent."""
        TorchAgent.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('Image Caption Model Arguments')
        # agent.add_argument('--embed_size', type=int , default=256,
        #                    help='dimension of word embedding vectors')
        # agent.add_argument('--hidden_size', type=int , default=512,
        #                    help='dimension of lstm hidden states')
        # agent.add_argument('--num_layers', type=int , default=1,
        #                    help='number of layers in lstm')
        # agent.add_argument('--max_pred_length', type=int, default=20,
        #                    help='maximum length of predicted caption in eval mode')
        # agent.add_argument('-lr', '--learning_rate', type=float, default=0.001,
        #                    help='learning rate')
        # agent.add_argument('-opt', '--optimizer', default='adam',
        #                    choices=['sgd', 'adam'],
        #                    help='Choose either sgd or adam as the optimizer.')
        # agent.add_argument('--use_feature_state', type='bool',
        #                    default=True,
        #                    help='Initialize LSTM state with image features')
        # agent.add_argument('--concat_img_feats', type='bool', default=True,
        #                    help='Concat resnet feats to each token during generation')
        VSEppAgent.dictionary_class().add_cmdline_args(argparser)

    @staticmethod
    def dictionary_class():
        return DictionaryAgent

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
