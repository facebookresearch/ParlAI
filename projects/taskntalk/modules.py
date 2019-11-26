#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch
from torch import nn


def xavier_init(module):
    """
    Xavier initializer for module parameters.
    """
    for parameter in module.parameters():
        if len(parameter.data.shape) == 1:
            # 1D vector means bias
            parameter.data.fill_(0)
        else:
            fan_in = parameter.data.size(0)
            fan_out = parameter.data.size(1)
            parameter.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))


class ImgNet(nn.Module):
    """
    Module to embed the visual information. Used by answerer agent. In ``forward``:
    Embed image attributes and concatenate them together.

    **Note:** ``parlai.core.image_featurizers.ImageLoader`` can also be
    used instead.
    """

    def __init__(self, feature_size, input_size=None):
        super().__init__()
        # input_size is needed for modules which require input_size specification
        # nn.Embedding requires input size to be specified, while nn.Conv2d doesn't
        self.net = nn.Embedding(input_size, feature_size)
        xavier_init(self)

    def forward(self, image):
        embeds = self.net(image)
        features = torch.cat(embeds.transpose(0, 1), 1)
        return features


class ListenNet(nn.Module):
    """
    Module for listening the sequence spoken by other agent.

    In ``forward``: Generate token embeddings.
    """

    def __init__(self, in_size, embed_size):
        super().__init__()
        self.net = nn.Embedding(in_size, embed_size)
        xavier_init(self)

    def forward(self, text_tokens):
        embeds = self.net(text_tokens)
        return embeds


class StateNet(nn.Module):
    """
    Module for containing the state update mechanism for an agent.

    In
    ``forward``: Update states by passing the embeddings through LSTMCell.
    """

    def __init__(self, embed_size, state_size):
        super().__init__()
        self.net = nn.LSTMCell(embed_size, state_size)
        xavier_init(self)

    def forward(self, states, embeds):
        states = self.net(embeds, states)
        return states


class SpeakNet(nn.Module):
    """
    Module for speaking a token based on current state.

    In ``forward``: Return a probability distribution of utterances of tokens.
    """

    def __init__(self, state_size, out_size):
        super().__init__()
        self.net = nn.Linear(state_size, out_size)
        self.softmax = nn.Softmax()
        xavier_init(self)

    def forward(self, state):
        out_distr = self.softmax(self.net(state))
        return out_distr


class PredictNet(nn.Module):
    """
    Module to make a prediction as per goal.

    Used by questioner agent. In
    ``forward``: Return a probability distribution of utterances of tokens.
    """

    def __init__(self, embed_size, state_size, out_size):
        super().__init__()
        self.net_lstm = nn.LSTMCell(embed_size, state_size)
        self.net_mlp = nn.Linear(state_size, out_size)
        self.softmax = nn.Softmax()
        xavier_init(self)

    def forward(self, task_embeds, states):
        states = self.net_lstm(task_embeds, states)
        out_distr = self.softmax(self.predict_net(states[1]))
        _, prediction = out_distr.max(1)
        return prediction
