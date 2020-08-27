#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from parlai.agents.seq2seq.modules import (
    OutputLayer,
    RNNEncoder,
    _transpose_hidden_state,
)
from parlai.core.torch_generator_agent import TorchGeneratorModel


class HredModel(TorchGeneratorModel):
    """
    HRED model that has a context LSTM layer.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        device,
        numlayers=2,
        dropout=0,
        bidirectional=False,
        rnn_class="lstm",
        lookuptable="unique",
        decoder="same",
        numsoftmax=1,
        attention="none",
        attention_length=48,
        attention_time="post",
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):

        super().__init__(
            padding_idx=padding_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            longest_label=longest_label,
        )

        rnn_class = nn.LSTM
        embedding_size = embeddingsize
        hidden_size = hiddensize

        self.encoder = HredEncoder(
            num_features=num_features,
            embeddingsize=embedding_size,
            hiddensize=hidden_size,
            rnn_class=rnn_class,
            numlayers=numlayers,
            device=device,
        )

        self.decoder = HredDecoder(
            num_features=num_features,
            embeddingsize=embedding_size,
            hiddensize=hidden_size,
            rnn_class=rnn_class,
            numlayers=numlayers,
        )

        self.output = OutputLayer(
            num_features=num_features,
            embeddingsize=embedding_size,
            hiddensize=hidden_size,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.
        """
        enc_out, hidden, attn_mask, context_vec = encoder_states
        # make sure we swap the hidden state around, apropos multigpu settings
        hidden = _transpose_hidden_state(hidden)

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if not torch.is_tensor(indices):
            # cast indices to a tensor if needed
            indices = torch.LongTensor(indices).to(hid.device)

        hid = hid.index_select(1, indices)
        if cell is None:
            hidden = hid
        else:
            cell = cell.index_select(1, indices)
            hidden = (hid, cell)

        # and bring it back to multigpu friendliness
        hidden = _transpose_hidden_state(hidden)
        context_vec = context_vec.index_select(0, indices)
        return enc_out, hidden, attn_mask, context_vec

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or vanilla rnn
            return torch.index_select(incremental_state, 0, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state
            )


class HredEncoder(RNNEncoder):
    """
    RNN Encoder.

    Modified to encode history vector in context lstm.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        device,
        padding_idx=0,
        rnn_class="lstm",
        numlayers=2,
        dropout=0.1,
        bidirectional=False,
        shared_lt=None,
        shared_rnn=None,
        input_dropout=0,
        unknown_idx=None,
        sparse=False,
    ):
        """
        Initialize recurrent encoder and context lstm.
        """
        super().__init__(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx,
            rnn_class,
            numlayers,
            dropout,
            bidirectional,
            shared_lt,
            shared_rnn,
            input_dropout,
            unknown_idx,
            sparse,
        )
        self.padding_idx = padding_idx
        self.context_lstm = nn.LSTM(
            hiddensize, hiddensize, numlayers, batch_first=True
        ).to(device)

    def forward(self, xs, context_vec, hist_lens):
        # encode current utterrance
        (enc_state, (hidden_state, cell_state), attn_mask) = super().forward(xs)
        # if all utterances in context vec length 1, unsqueeze to prevent loss of dimensionality
        if len(context_vec.shape) < 2:
            context_vec = context_vec.unsqueeze(1)
        # get utt lengths of each utt in context vector
        utt_lens = torch.sum(context_vec.ne(0).int(), dim=1)
        # sort by lengths descending for utterance encoder
        sorted_lens, sorted_idx = utt_lens.sort(descending=True)
        sorted_context_vec = context_vec[sorted_idx]
        (_, (sorted_hidden_state, _), _) = super().forward(sorted_context_vec)
        sorted_final_hidden_states = sorted_hidden_state[:, -1, :]

        ### reshape and pad hidden states to bsz x max_hist_len x hidden_size using hist_lens
        original_order_final_hidden = torch.zeros_like(
            sorted_final_hidden_states
        ).scatter_(
            0,
            sorted_idx.unsqueeze(1).expand(-1, sorted_final_hidden_states.shape[1]),
            sorted_final_hidden_states,
        )
        # pad to max hist_len
        original_size_final_hidden = self.sequence_to_padding(
            original_order_final_hidden, hist_lens
        )
        # pack padded sequence so that we ignore padding
        original_size_final_hidden_packed = nn.utils.rnn.pack_padded_sequence(
            original_size_final_hidden,
            hist_lens.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        # pass through context lstm
        _, (context_h_n, _) = self.context_lstm(original_size_final_hidden_packed)
        return (
            enc_state,
            (hidden_state, cell_state),
            attn_mask,
            _transpose_hidden_state(context_h_n),
        )

    def sequence_to_padding(self, x, lengths):
        """
        Return padded and reshaped sequence (x) according to tensor lengths
        Example:
            x = tensor([[1, 2], [2, 3], [4, 0], [5, 6], [7, 8], [9, 10]])
            lengths = tensor([1, 2, 2, 1])
        Would output:
            tensor([[[1, 2], [0, 0]],
                    [[2, 3], [4, 0]],
                    [[5, 6], [7, 8]],
                    [[9, 10], [0, 0]]])
        """
        ret_tensor = torch.zeros(
            (lengths.shape[0], torch.max(lengths).int()) + tuple(x.shape[1:])
        ).to(x.device)
        cum_len = 0
        for i, l in enumerate(lengths):
            ret_tensor[i, :l] = x[cum_len : cum_len + l]
            cum_len += l
        return ret_tensor


class HredDecoder(nn.Module):
    """
    Recurrent decoder module that uses dialog history encoded by context lstm.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        padding_idx=0,
        rnn_class="lstm",
        numlayers=2,
        dropout=0.1,
        bidir_input=False,
        attn_length=-1,
        sparse=False,
    ):
        """
        Initialize recurrent decoder.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.hsz = hiddensize
        self.esz = embeddingsize

        self.lt = nn.Embedding(
            num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse
        )
        self.rnn = rnn_class(
            embeddingsize + hiddensize,
            hiddensize,
            numlayers,
            dropout=dropout if numlayers > 1 else 0,
            batch_first=True,
        )

    def forward(self, xs, encoder_output, incremental_state=None):
        """
        Decode from input tokens.

        :param xs: (bsz x seqlen) LongTensor of input token indices
        :param encoder_output: output from HredEncoder. Tuple containing
            (enc_out, enc_hidden, attn_mask, context_hidden) tuple.
        :param incremental_state: most recent hidden state to the decoder.
        :returns: (output, hidden_state) pair from the RNN.
            - output is a bsz x time x latentdim matrix. This value must be passed to
                the model's OutputLayer for a final softmax.
            - hidden_state depends on the choice of RNN
        """
        (
            enc_state,
            (hidden_state, cell_state),
            attn_mask,
            context_hidden,
        ) = encoder_output

        # sequence indices => sequence embeddings
        seqlen = xs.size(1)
        xes = self.dropout(self.lt(xs))

        # concatentate context lstm hidden state
        context_hidden_final_layer = context_hidden[:, -1, :].unsqueeze(1)
        resized_context_h = context_hidden_final_layer.expand(-1, seqlen, -1)
        xes = torch.cat((xes, resized_context_h), dim=-1).to(xes.device)

        # run through rnn with None as initial decoder state
        # source for zeroes hidden state: http://www.cs.toronto.edu/~lcharlin/papers/vhred_aaai17.pdf
        output, new_hidden = self.rnn(xes, None)

        return output, _transpose_hidden_state(new_hidden)
