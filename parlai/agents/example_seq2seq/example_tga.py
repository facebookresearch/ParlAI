#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Example TorchGeneratorAgent model.

This demonstrates the minimal structure of a building a generative model.

You can train this agent with:

.. code-block:: python

    python examples/train_model.py -mf /tmp/example_model -t convai2 -m example_tga -bs 16
"""  # noqa: E501

import torch
import torch.nn as nn
import torch.nn.functional as F
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

    def forward(self, input_tokens):
        """
        Perform the forward pass for the encoder.

        Input *must* be input_tokens, which are the context tokens given
        as a matrix of lookup IDs.

        :param input_tokens:
            Input tokens as a bsz x seqlen LongTensor.
            Likely will contain padding.

        :return:
            You can return anything you like; it is will be passed verbatim
            into the decoder for conditioning. However, it should be something
            you can easily manipulate in ``reorder_encoder_states``.
            This particular implementation returns the output tensor from the LSTM.
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
        _vocab_size, self.esz = embeddings.shape
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=self.esz,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.lin_out = nn.Linear(hidden_size, self.esz)

    def forward(self, input, encoder_state, incr_state=None):
        """
        Run forward pass.

        :param input:
            The currently generated tokens from the decoder.
        :param encoder_state:
            The output from the encoder module.
        :parm incr_state:
            The previous hidden state of the decoder.
        """
        embedded = self.embeddings(input)
        # In the next example, we'll keep the (h, c) around for fast decoding
        output, _ = self.lstm(embedded)

        # we should return the decoder output, before the final softmax
        # we can additionally provide some incremental state. This version won't
        # use incremental state, so we'll just use None
        incremental_state = None
        return output, incremental_state


class ExampleModel(tga.TorchGeneratorModel):
    """
    ExampleModel implements the final components of TorchGeneratorModel.

    Mainly needs to implement reorder_encoder_size, and instantiate
    self.encoder and self.decoder.
    """

    def __init__(self, dictionary, esz=256, hidden_size=1024):
        super().__init__(
            padding_idx=dictionary[dictionary.pad_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token]
        )
        self.embeddings = nn.Embedding(len(dictionary), esz)
        self.encoder = Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder shapes to select only the given batch indices.

        Since encoder_state can be arbitrary, you must implement this yourself.
        Typically you will just want to index select on the batch dimension.
        """
        return torch.index_select(encoder_states, indices)

    def output(self, decoder_output):
        """Perform the final output -> logits transformation."""
        return F.linear(decoder_output, self.embeddings.weight)


class ExampleSeqAgent(tga.TorchGeneratorAgent):
    """
    Example agent.

    Implements the interface for TorchGeneratorAgent. The minimum requirement
    is that it implements ``build_model``.
    """

    def build_model(self):
        """Construct the model."""
        self.model = ExampleModel(self.dict, self.opt['hidden_size'])
