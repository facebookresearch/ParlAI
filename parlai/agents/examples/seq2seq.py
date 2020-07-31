#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""
Example TorchGeneratorAgent model.

This demonstrates the minimal structure of a building a generative model, consisting of
an encoder and decoder that each contain a 1-layer LSTM. The subclassed model and agent
handle passing in the current decoder state during incremental decoding, as well as
reordering the encoder/decoder states. The base TorchGeneratorAgent class handles common
generator features like forced decoding, beam search, n-gram beam blocking, and top-k
and top-p/nucleus sampling.

You can train this agent to a reasonable accuracy with:

.. code-block:: bash

    parlai train_model -m examples/seq2seq \
        -mf /tmp/example_model \
        -t convai2 -bs 32 -eps 2 --truncate 128

Afterwards, you can play with --beam-size to see how responses differ with
different beam lengths.
"""  # noqa: E501

import torch.nn as nn
import torch.nn.functional as F
import parlai.core.torch_generator_agent as tga


class Encoder(nn.Module):
    """
    Example encoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size.

    Pay particular attention to the ``forward`` output.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.

        Arguments here can be used to provide hyperparameters.
        """
        # must call super on all nn.Modules.
        super().__init__()

        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
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
            This particular implementation returns the hidden and cell states from the
            LSTM.
        """
        embedded = self.embeddings(input_tokens)
        _output, hidden = self.lstm(embedded)
        return hidden


class Decoder(nn.Module):
    """
    Basic example decoder, consisting of an embedding layer and a 1-layer LSTM with the
    specified hidden size. Decoder allows for incremental decoding by ingesting the
    current incremental state on each forward pass.

    Pay particular note to the ``forward``.
    """

    def __init__(self, embeddings, hidden_size):
        """
        Initialization.

        Arguments here can be used to provide hyperparameters.
        """
        super().__init__()
        self.embeddings = embeddings
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )

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
        if incr_state is None:
            # this is our very first call. We want to seed the LSTM with the
            # hidden state of the decoder
            state = encoder_state
        else:
            # We've generated some tokens already, so we can reuse the existing
            # decoder state
            state = incr_state

        # get the new output and decoder incremental state
        output, incr_state = self.lstm(embedded, state)

        return output, incr_state


class ExampleModel(tga.TorchGeneratorModel):
    """
    ExampleModel implements the abstract methods of TorchGeneratorModel to define how to
    re-order encoder states and decoder incremental states.

    It also instantiates the embedding table, encoder, and decoder, and defines the
    final output layer.
    """

    def __init__(self, dictionary, hidden_size=1024):
        super().__init__(
            padding_idx=dictionary[dictionary.null_token],
            start_idx=dictionary[dictionary.start_token],
            end_idx=dictionary[dictionary.end_token],
            unknown_idx=dictionary[dictionary.unk_token],
        )
        self.embeddings = nn.Embedding(len(dictionary), hidden_size)
        self.encoder = Encoder(self.embeddings, hidden_size)
        self.decoder = Decoder(self.embeddings, hidden_size)

    def output(self, decoder_output):
        """
        Perform the final output -> logits transformation.
        """
        return F.linear(decoder_output, self.embeddings.weight)

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder the encoder states to select only the given batch indices.

        Since encoder_state can be arbitrary, you must implement this yourself.
        Typically you will just want to index select on the batch dimension.
        """
        h, c = encoder_states
        return h[:, indices, :], c[:, indices, :]

    def reorder_decoder_incremental_state(self, incr_state, indices):
        """
        Reorder the decoder states to select only the given batch indices.

        This method can be a stub which always returns None; this will result in the
        decoder doing a complete forward pass for every single token, making generation
        O(n^2). However, if any state can be cached, then this method should be
        implemented to reduce the generation complexity to O(n).
        """
        h, c = incr_state
        return h[:, indices, :], c[:, indices, :]


class Seq2seqAgent(tga.TorchGeneratorAgent):
    """
    Example agent.

    Implements the interface for TorchGeneratorAgent. The minimum requirement is that it
    implements ``build_model``, but we will want to include additional command line
    parameters.
    """

    @classmethod
    def add_cmdline_args(cls, argparser):
        """
        Add CLI arguments.
        """
        # Make sure to add all of TorchGeneratorAgent's arguments
        super(Seq2seqAgent, cls).add_cmdline_args(argparser)

        # Add custom arguments only for this model.
        group = argparser.add_argument_group('Example TGA Agent')
        group.add_argument(
            '-hid', '--hidden-size', type=int, default=1024, help='Hidden size.'
        )

    def build_model(self):
        """
        Construct the model.
        """

        model = ExampleModel(self.dict, self.opt['hidden_size'])
        # Optionally initialize pre-trained embeddings by copying them from another
        # source: GloVe, fastText, etc.
        self._copy_embeddings(model.embeddings.weight, self.opt['embedding_type'])
        return model
