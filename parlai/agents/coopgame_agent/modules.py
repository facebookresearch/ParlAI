# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

"""Set of modules essential for agents in cooperative and goal-based
conversational games. Agents may have information about visual content
(image, video or image attributes). These agents use following modules,
each module can be subclassed and replaced in agent according to need.
"""
import math
import torch
from torch import nn


def xavier_init(module):
    """Xavier initializer for module parameters."""
    for parameter in module.parameters():
        if len(parameter.data.shape) == 1:
            # 1D vector means bias
            parameter.data.fill_(0)
        else:
            fan_in = parameter.data.size(0)
            fan_out = parameter.data.size(1)
            parameter.data.normal_(0, math.sqrt(2 / (fan_in + fan_out)))


class ImgNet(nn.Module):
    """Module to embed the visual information. Usually, QBot is blindfolded,
    so this exists only in ABot.
    """
    def __init__(self, feature_size, input_size=None):
        super().__init__()
        # input_size is needed for modules which require input_size specification
        # nn.Embedding requires input size to be specified, while nn.Conv2d doesn't
        self.net = nn.Embedding(input_size, feature_size)
        xavier_init(self)

    def forward(self, image):
        """Embed image attributes and concatenate them together."""
        embeds = self.net(image)
        features = torch.cat(embeds.transpose(0, 1), 1)
        return features


class ListenNet(nn.Module):
    """Module for listening the sequence spoken by other agent."""
    def __init__(self, in_size, embed_size):
        super().__init__()
        self.net = nn.Embedding(in_size, embed_size)
        xavier_init(self)

    def forward(self, text_tokens):
        """Generate token embeddings."""
        embeds = self.net(text_tokens)
        return embeds


class StateNet(nn.Module):
    """Module for containing the state update mechanism for an agent."""
    def __init__(self, embed_size, state_size):
        super().__init__()
        self.net = nn.LSTMCell(embed_size, state_size)
        xavier_init(self)

    def forward(self, states, embeds):
        """Update states by passing the embeddings through LSTMCell."""
        states = self.net(embeds, states)
        return states


class SpeakNet(nn.Module):
    """Module for speaking a token based on current state."""
    def __init__(self, state_size, out_size):
        super().__init__()
        self.net = nn.Linear(state_size, out_size)
        self.softmax = nn.Softmax()
        xavier_init(self)

    def forward(self, state):
        """Return a probability distribution of utterances of tokens."""
        out_distr = self.softmax(self.net(state))
        return out_distr


class PredictNet(nn.Module):
    """Module to make a prediction as per goal. Usually QBot is assigned
    a task to be performed at the end of dialog episode.
    """
    def __init__(self, embed_size, state_size, out_size):
        super().__init__()
        self.net_lstm = nn.LSTMCell(embed_size, state_size)
        self.net_mlp = nn.Linear(state_size, out_size)
        self.softmax = nn.Softmax()
        xavier_init(self)

    def forward(self, task_embeds, states):
        """Return a probability distribution of utterances of tokens."""
        states = self.net_lstm(task_embeds, states)
        out_distr = self.softmax(self.predict_net(states[1]))
        _, prediction = out_distr.max(1)
        return prediction
