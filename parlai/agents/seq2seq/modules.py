#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Module files as torch.nn.Module subclasses for Seq2seqAgent.
"""

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F

from parlai.utils.torch import neginf
from parlai.core.torch_generator_agent import TorchGeneratorModel


def _transpose_hidden_state(hidden_state):
    """
    Transpose the hidden state so that batch is the first dimension.

    RNN modules produce (num_layers x batchsize x dim) hidden state, but DataParallel
    expects batch size to be first. This helper is used to ensure that we're always
    outputting batch-first, in case DataParallel tries to stitch things back together.
    """
    if isinstance(hidden_state, tuple):
        return tuple(map(_transpose_hidden_state, hidden_state))
    elif torch.is_tensor(hidden_state):
        return hidden_state.transpose(0, 1)
    else:
        raise ValueError("Don't know how to transpose {}".format(hidden_state))


def opt_to_kwargs(opt):
    """
    Get kwargs for seq2seq from opt.
    """
    kwargs = {}
    for k in [
        'numlayers',
        'dropout',
        'bidirectional',
        'rnn_class',
        'lookuptable',
        'decoder',
        'numsoftmax',
        'attention',
        'attention_length',
        'attention_time',
        'input_dropout',
    ]:
        if k in opt:
            kwargs[k] = opt[k]
    return kwargs


class Seq2seq(TorchGeneratorModel):
    """
    Sequence to sequence parent module.
    """

    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        numlayers=2,
        dropout=0,
        bidirectional=False,
        rnn_class='lstm',
        lookuptable='unique',
        decoder='same',
        numsoftmax=1,
        attention='none',
        attention_length=48,
        attention_time='post',
        padding_idx=0,
        start_idx=1,
        end_idx=2,
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):
        """
        Initialize seq2seq model.

        See cmdline args in Seq2seqAgent for description of arguments.
        """
        super().__init__(
            padding_idx=padding_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
            longest_label=longest_label,
        )
        self.attn_type = attention

        rnn_class = Seq2seq.RNN_OPTS[rnn_class]
        self.decoder = RNNDecoder(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx=padding_idx,
            rnn_class=rnn_class,
            numlayers=numlayers,
            dropout=dropout,
            attn_type=attention,
            attn_length=attention_length,
            attn_time=attention_time,
            bidir_input=bidirectional,
        )

        shared_lt = (
            self.decoder.lt  # share embeddings between rnns
            if lookuptable in ('enc_dec', 'all')
            else None
        )
        shared_rnn = self.decoder.rnn if decoder == 'shared' else None
        self.encoder = RNNEncoder(
            num_features,
            embeddingsize,
            hiddensize,
            padding_idx=padding_idx,
            rnn_class=rnn_class,
            numlayers=numlayers,
            dropout=dropout,
            bidirectional=bidirectional,
            shared_lt=shared_lt,
            shared_rnn=shared_rnn,
            unknown_idx=unknown_idx,
            input_dropout=input_dropout,
        )

        shared_weight = (
            self.decoder.lt  # use embeddings for projection
            if lookuptable in ('dec_out', 'all')
            else None
        )
        self.output = OutputLayer(
            num_features,
            embeddingsize,
            hiddensize,
            dropout=dropout,
            numsoftmax=numsoftmax,
            shared_weight=shared_weight,
            padding_idx=padding_idx,
        )

    def reorder_encoder_states(self, encoder_states, indices):
        """
        Reorder encoder states according to a new set of indices.
        """
        enc_out, hidden, attn_mask = encoder_states

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

        if self.attn_type != 'none':
            enc_out = enc_out.index_select(0, indices)
            attn_mask = attn_mask.index_select(0, indices)

        # and bring it back to multigpu friendliness
        hidden = _transpose_hidden_state(hidden)

        return enc_out, hidden, attn_mask

    def reorder_decoder_incremental_state(self, incremental_state, inds):
        if torch.is_tensor(incremental_state):
            # gru or vanilla rnn
            return torch.index_select(incremental_state, 0, inds).contiguous()
        elif isinstance(incremental_state, tuple):
            return tuple(
                self.reorder_decoder_incremental_state(x, inds)
                for x in incremental_state
            )


class UnknownDropout(nn.Module):
    """
    With set frequency, replaces tokens with unknown token.

    This layer can be used right before an embedding layer to make the model more robust
    to unknown words at test time.
    """

    def __init__(self, unknown_idx, probability):
        """
        Initialize layer.

        :param unknown_idx: index of unknown token, replace tokens with this
        :param probability: during training, replaces tokens with unknown token
                            at this rate.
        """
        super().__init__()
        self.unknown_idx = unknown_idx
        self.prob = probability

    def forward(self, input):
        """
        If training and dropout rate > 0, masks input with unknown token.
        """
        if self.training and self.prob > 0:
            mask = input.new(input.size()).float().uniform_(0, 1) < self.prob
            input.masked_fill_(mask, self.unknown_idx)
        return input


class RNNEncoder(nn.Module):
    """
    RNN Encoder.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        padding_idx=0,
        rnn_class='lstm',
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
        Initialize recurrent encoder.
        """
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = numlayers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hiddensize

        if input_dropout > 0 and unknown_idx is None:
            raise RuntimeError('input_dropout > 0 but unknown_idx not set')
        self.input_dropout = UnknownDropout(unknown_idx, input_dropout)

        if shared_lt is None:
            self.lt = nn.Embedding(
                num_features, embeddingsize, padding_idx=padding_idx, sparse=sparse
            )
        else:
            self.lt = shared_lt

        if shared_rnn is None:
            self.rnn = rnn_class(
                embeddingsize,
                hiddensize,
                numlayers,
                dropout=dropout if numlayers > 1 else 0,
                batch_first=True,
                bidirectional=bidirectional,
            )
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def forward(self, xs):
        """
        Encode sequence.

        :param xs: (bsz x seqlen) LongTensor of input token indices

        :returns: encoder outputs, hidden state, attention mask
            encoder outputs are the output state at each step of the encoding.
            the hidden state is the final hidden state of the encoder.
            the attention mask is a mask of which input values are nonzero.
        """
        bsz = len(xs)

        # embed input tokens
        xs = self.input_dropout(xs)
        xes = self.dropout(self.lt(xs))
        attn_mask = xs.ne(0)
        try:
            x_lens = torch.sum(attn_mask.int(), dim=1).cpu()
            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        if packed:
            # total_length to make sure we give the proper length in the case
            # of multigpu settings.
            # https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
            encoder_output, _ = pad_packed_sequence(
                encoder_output, batch_first=True, total_length=xs.size(1)
            )

        if self.dirs > 1:
            # project to decoder dimension by taking sum of forward and back
            if isinstance(self.rnn, nn.LSTM):
                hidden = (
                    hidden[0].view(-1, self.dirs, bsz, self.hsz).sum(1),
                    hidden[1].view(-1, self.dirs, bsz, self.hsz).sum(1),
                )
            else:
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).sum(1)

        return encoder_output, _transpose_hidden_state(hidden), attn_mask


class RNNDecoder(nn.Module):
    """
    Recurrent decoder module.

    Can be used as a standalone language model or paired with an encoder.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        padding_idx=0,
        rnn_class='lstm',
        numlayers=2,
        dropout=0.1,
        bidir_input=False,
        attn_type='none',
        attn_time='pre',
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
            embeddingsize,
            hiddensize,
            numlayers,
            dropout=dropout if numlayers > 1 else 0,
            batch_first=True,
        )

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(
            attn_type=attn_type,
            hiddensize=hiddensize,
            embeddingsize=embeddingsize,
            bidirectional=bidir_input,
            attn_length=attn_length,
            attn_time=attn_time,
        )

    def forward(self, xs, encoder_output, incremental_state=None):
        """
        Decode from input tokens.

        :param xs: (bsz x seqlen) LongTensor of input token indices
        :param encoder_output: output from RNNEncoder. Tuple containing
            (enc_out, enc_hidden, attn_mask) tuple.
        :param incremental_state: most recent hidden state to the decoder.
            If None, the hidden state of the encoder is used as initial state,
            and the full sequence is computed. If not None, computes only the
            next forward in the sequence.

        :returns: (output, hidden_state) pair from the RNN.

            - output is a bsz x time x latentdim matrix. If incremental_state is
                given, the time dimension will be 1. This value must be passed to
                the model's OutputLayer for a final softmax.
            - hidden_state depends on the choice of RNN
        """
        enc_state, enc_hidden, attn_mask = encoder_output
        # in case of multi gpu, we need to transpose back out the hidden state
        attn_params = (enc_state, attn_mask)

        if incremental_state is not None:
            # we're doing it piece by piece, so we have a more important hidden
            # seed, and we only need to compute for the final timestep
            hidden = _transpose_hidden_state(incremental_state)
            # only need the last timestep then
            xs = xs[:, -1:]
        else:
            # starting fresh, or generating from scratch. Use the encoder hidden
            # state as our start state
            hidden = _transpose_hidden_state(enc_hidden)

        if isinstance(hidden, tuple):
            hidden = tuple(x.contiguous() for x in hidden)
        else:
            hidden = hidden.contiguous()

        # sequence indices => sequence embeddings
        seqlen = xs.size(1)
        xes = self.dropout(self.lt(xs))

        if self.attn_time == 'pre':
            # modify input vectors with attention
            # attention module requires we do this one step at a time
            new_xes = []
            for i in range(seqlen):
                nx, _ = self.attention(xes[:, i : i + 1], hidden, attn_params)
                new_xes.append(nx)
            xes = torch.cat(new_xes, 1).to(xes.device)

        if self.attn_time != 'post':
            # no attn, we can just trust the rnn to run through
            output, new_hidden = self.rnn(xes, hidden)
        else:
            # uh oh, post attn, we need run through one at a time, and do the
            # attention modifications
            new_hidden = hidden
            output = []
            for i in range(seqlen):
                o, new_hidden = self.rnn(xes[:, i, :].unsqueeze(1), new_hidden)
                o, _ = self.attention(o, new_hidden, attn_params)
                output.append(o)
            output = torch.cat(output, dim=1).to(xes.device)

        return output, _transpose_hidden_state(new_hidden)


class Identity(nn.Module):
    def forward(self, x):
        return x


class OutputLayer(nn.Module):
    """
    Takes in final states and returns distribution over candidates.
    """

    def __init__(
        self,
        num_features,
        embeddingsize,
        hiddensize,
        dropout=0,
        numsoftmax=1,
        shared_weight=None,
        padding_idx=-1,
    ):
        """
        Initialize output layer.

        :param num_features:  number of candidates to rank
        :param hiddensize:    (last) dimension of the input vectors
        :param embeddingsize: (last) dimension of the candidate vectors
        :param numsoftmax:   (default 1) number of softmaxes to calculate.
                              see arxiv.org/abs/1711.03953 for more info.
                              increasing this slows down computation but can
                              add more expressivity to the embeddings.
        :param shared_weight: (num_features x esz) vector of weights to use as
                              the final linear layer's weight matrix. default
                              None starts with a new linear layer.
        :param padding_idx:   model should output a large negative number for
                              score at this index. if set to -1 (default),
                              this is disabled. if >= 0, subtracts one from
                              num_features and always outputs -1e20 at this
                              index. only used when shared_weight is not None.
                              setting this param helps protect gradient from
                              entering shared embedding matrices.
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        self.padding_idx = padding_idx
        rng = 1.0 / math.sqrt(num_features)
        self.bias = Parameter(torch.Tensor(num_features).uniform_(-rng, rng))

        # embedding to scores
        if shared_weight is None:
            # just a regular linear layer
            self.shared = False
            self.weight = Parameter(
                torch.Tensor(num_features, embeddingsize).normal_(0, 1)
            )
        else:
            # use shared weights and a bias layer instead
            self.shared = True
            self.weight = shared_weight.weight

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.esz = embeddingsize
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hiddensize, numsoftmax, bias=False)
            self.latent = nn.Linear(hiddensize, numsoftmax * embeddingsize)
            self.activation = nn.Tanh()
        else:
            # rnn output to embedding
            if hiddensize != embeddingsize:
                # learn projection to correct dimensions
                self.o2e = nn.Linear(hiddensize, embeddingsize, bias=True)
            else:
                # no need for any transformation here
                self.o2e = Identity()

    def forward(self, input):
        """
        Compute scores from inputs.

        :param input: (bsz x seq_len x num_directions * hiddensize) tensor of
                       states, e.g. the output states of an RNN

        :returns: (bsz x seqlen x num_cands) scores for each candidate
        """
        # next compute scores over dictionary
        if self.numsoftmax > 1:
            bsz = input.size(0)
            seqlen = input.size(1) if input.dim() > 1 else 1

            # first compute different softmax scores based on input vec
            # hsz => numsoftmax * esz
            latent = self.latent(input)
            active = self.dropout(self.activation(latent))
            # esz => num_features
            logit = F.linear(active.view(-1, self.esz), self.weight, self.bias)

            # calculate priors: distribution over which softmax scores to use
            # hsz => numsoftmax
            prior_logit = self.prior(input).view(-1, self.numsoftmax)
            # softmax over numsoftmax's
            prior = self.softmax(prior_logit)

            # now combine priors with logits
            prob = self.softmax(logit).view(bsz * seqlen, self.numsoftmax, -1)
            probs = (prob * prior.unsqueeze(2)).sum(1).view(bsz, seqlen, -1)
            scores = probs.log()
        else:
            # hsz => esz, good time for dropout
            e = self.dropout(self.o2e(input))
            # esz => num_features
            scores = F.linear(e, self.weight, self.bias)

        if self.padding_idx >= 0:
            scores[:, :, self.padding_idx] = neginf(scores.dtype)

        return scores


class AttentionLayer(nn.Module):
    """
    Computes attention between hidden and encoder states.

    See arxiv.org/abs/1508.04025 for more info on each attention type.
    """

    def __init__(
        self,
        attn_type,
        hiddensize,
        embeddingsize,
        bidirectional=False,
        attn_length=-1,
        attn_time='pre',
    ):
        """
        Initialize attention layer.
        """
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hiddensize
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = embeddingsize
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')

            # linear layer for combining applied attention weights with input
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim, bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + input_dim, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, attn_params):
        """
        Compute attention over attn_params given input and hidden states.

        :param xes:         input state. will be combined with applied
                            attention.
        :param hidden:      hidden state from model. will be used to select
                            states to attend to in from the attn_params.
        :param attn_params: tuple of encoder output states and a mask showing
                            which input indices are nonzero.

        :returns: output, attn_weights
                  output is a new state of same size as input state `xes`.
                  attn_weights are the weights given to each state in the
                  encoder outputs.
        """
        if self.attention == 'none':
            # do nothing, no attention
            return xes, None

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        enc_out, attn_mask = attn_params
        bsz, seqlen, hszXnumdir = enc_out.size()
        numlayersXnumdir = last_hidden.size(1)

        if self.attention == 'local':
            # local attention weights aren't based on encoder states
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)

            # adjust state sizes to the fixed window size
            if seqlen > self.max_length:
                offset = seqlen - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
                seqlen = self.max_length
            if attn_weights.size(1) > seqlen:
                attn_weights = attn_weights.narrow(1, 0, seqlen)
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                # concat hidden state and encoder outputs
                hid = hid.expand(bsz, seqlen, numlayersXnumdir)
                h_merged = torch.cat((enc_out, hid), 2)
                # then do linear combination of them with activation
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                # dot product between hidden and encoder outputs
                if numlayersXnumdir != hszXnumdir:
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)
            elif self.attention == 'general':
                # before doing dot product, transform hidden state with linear
                # same as dot if linear is identity
                hid = self.attn(hid)
                enc_t = enc_out.transpose(1, 2)
                attn_w_premask = torch.bmm(hid, enc_t).squeeze(1)

            # calculate activation scores, apply mask if needed
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask.masked_fill_((~attn_mask), neginf(attn_w_premask.dtype))
            attn_weights = F.softmax(attn_w_premask, dim=1)

        # apply the attention weights to the encoder states
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        # concatenate the input and encoder states
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        # combine them with a linear layer and tanh activation
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))

        return output, attn_weights
