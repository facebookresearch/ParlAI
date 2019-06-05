#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example TorchGeneratorAgent model.

This demonstrates the minimal structure of a building a generative model.
"""
from typing import Any

import torch
import torch.nn as nn
import parlai.core.torch_generator_agent as tga


class Encoder(nn.Module):
    """
    Example encoder.

    Pay particular attention to the ``forward`` output.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.

        Arguments here can be used to provide hyperparameters.
        """
        # must call super on all nn.Modules.
        super().__init__()

        _vocab_size, esz = embeddings.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=esz,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )

    def forward(self, input_tokens: torch.LongTensor):
        """
        Perform the forward pass for the encoder.

        Input *must* be input_tokens, which are the context tokens given
        as a matrix of lookup IDs.

        :param input_tokens:
            Input tokens as a bsz x seqlen LongTensor.
            Likely will contain padding.

        :return:
            This particular implementation returns a regular Tensor
            containing the hidden states. However, your model can return
            any type here, and it will be directly passed to the model.
        """
        embedded = self.embeddings(input_tokens)
        hidden = self.lstm(embedded)
        return hidden


class Decoder(nn.Module):
    """
    Basic example Decoder.

    Pay particular note to the ``forward``.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.

        Arguments here can be used to provide hyperparameters.
        """
        super().__init__()
        _vocab_size, esz = embeddings.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=esz,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.transformation = nn.Linear(hidden_size, esz)

    def forward(self, input: torch.LongTensor, encoder_state: Any, incr_state: Any = None):
        """
        :param input:
            The currently generated tokens from the decoder
        """


class ExampleModel(tga.TorchGeneratorModel):
    def __init__(self, dictionary, hidden_size=1024):
        # Beam search needs to be 
        super().__init__(
            padding_idx=dictionary[dictionary.pad_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token]
        )

    def reorder_encoder_states(self, encoder_states, indicies):
        pass


class ExampleSeqAgent(tga.TorchGeneratorAgent):
    """
    Example agent.

    Implements the interface for TorchGeneratorAgent. The minimum requirement
    is that it implements ``build_model``.
    """

    def build_model(self):
        self.model = ExampleModel(self.dict, self.opt['hidden_size'])
