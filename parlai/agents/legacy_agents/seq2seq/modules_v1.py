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

from parlai.utils.torch import NEAR_INF


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


def pad(tensor, length, dim=0, pad=0):
    """
    Pad tensor to a specific length.

    :param tensor: vector to pad
    :param length: new length
    :param dim: (default 0) dimension to pad

    :returns: padded tensor if the tensor is shorter than length
    """
    if tensor.size(dim) < length:
        return torch.cat(
            [
                tensor,
                tensor.new(
                    *tensor.size()[:dim],
                    length - tensor.size(dim),
                    *tensor.size()[dim + 1 :],
                ).fill_(pad),
            ],
            dim=dim,
        )
    else:
        return tensor


class Seq2seq(nn.Module):
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
        unknown_idx=3,
        input_dropout=0,
        longest_label=1,
    ):
        """
        Initialize seq2seq model.

        See cmdline args in Seq2seqAgent for description of arguments.
        """
        super().__init__()
        self.attn_type = attention

        self.NULL_IDX = padding_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

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
            self.decoder.lt.weight  # use embeddings for projection
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

    def _encode(self, xs, prev_enc=None):
        """
        Encode the input or return cached encoder state.
        """
        if prev_enc is not None:
            return prev_enc
        else:
            return self.encoder(xs)

    def _starts(self, bsz):
        """
        Return bsz start tokens.
        """
        return self.START.detach().expand(bsz, 1)

    def _decode_forced(self, ys, encoder_states):
        """
        Decode with teacher forcing.
        """
        bsz = ys.size(0)
        seqlen = ys.size(1)

        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])

        # input to model is START + each target except the last
        y_in = ys.narrow(1, 0, seqlen - 1)
        xs = torch.cat([self._starts(bsz), y_in], 1)

        scores = []
        if self.attn_type == 'none':
            # do the whole thing in one go
            output, hidden = self.decoder(xs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
        else:
            # need to feed in one token at a time so we can do attention
            # TODO: do we need to do this? actually shouldn't need to since we
            # don't do input feeding
            for i in range(seqlen):
                xi = xs.select(1, i).unsqueeze(1)
                output, hidden = self.decoder(xi, hidden, attn_params)
                score = self.output(output)
                scores.append(score)

        scores = torch.cat(scores, 1)
        return scores

    def _decode(self, encoder_states, maxlen):
        """
        Decode maxlen tokens.
        """
        hidden = encoder_states[1]
        attn_params = (encoder_states[0], encoder_states[2])
        bsz = encoder_states[0].size(0)

        xs = self._starts(bsz)  # input start token

        scores = []
        for _ in range(maxlen):
            # generate at most longest_label tokens
            output, hidden = self.decoder(xs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
            xs = score.max(2)[1]  # next input is current predicted output

        scores = torch.cat(scores, 1)
        return scores

    def _align_inds(self, encoder_states, cand_inds):
        """
        Select the encoder states relevant to valid candidates.
        """
        enc_out, hidden, attn_mask = encoder_states

        # LSTM or GRU/RNN hidden state?
        if isinstance(hidden, torch.Tensor):
            hid, cell = hidden, None
        else:
            hid, cell = hidden

        if len(cand_inds) != hid.size(1):
            # if the number of candidates is mismatched from the number of
            # hidden states, we throw out the hidden states we won't rank with
            cand_indices = hid.new(cand_inds)
            hid = hid.index_select(1, cand_indices)
            if cell is None:
                hidden = hid
            else:
                cell = cell.index_select(1, cand_indices)
                hidden = (hid, cell)

            if self.attn_type != 'none':
                enc_out = enc_out.index_select(0, cand_indices)
                attn_mask = attn_mask.index_select(0, cand_indices)

        return enc_out, hidden, attn_mask

    def _extract_cur(self, encoder_states, index, num_cands):
        """
        Extract encoder states at current index and expand them.
        """
        enc_out, hidden, attn_mask = encoder_states
        if isinstance(hidden, torch.Tensor):
            cur_hid = hidden.select(1, index).unsqueeze(1).expand(-1, num_cands, -1)
        else:
            cur_hid = (
                hidden[0]
                .select(1, index)
                .unsqueeze(1)
                .expand(-1, num_cands, -1)
                .contiguous(),
                hidden[1]
                .select(1, index)
                .unsqueeze(1)
                .expand(-1, num_cands, -1)
                .contiguous(),
            )

        cur_enc, cur_mask = None, None
        if self.attn_type != 'none':
            cur_enc = enc_out[index].unsqueeze(0).expand(num_cands, -1, -1)
            cur_mask = attn_mask[index].unsqueeze(0).expand(num_cands, -1)
        return cur_enc, cur_hid, cur_mask

    def _rank(self, cands, cand_inds, encoder_states):
        """
        Rank each cand by the average log-probability of the sequence.
        """
        if cands is None:
            return None
        encoder_states = self._align_inds(encoder_states, cand_inds)

        cand_scores = []
        for batch_idx in range(len(cands)):
            # we do one set of candidates at a time
            curr_cs = cands[batch_idx]
            num_cands = curr_cs.size(0)

            # select just the one hidden state
            cur_enc_states = self._extract_cur(encoder_states, batch_idx, num_cands)

            score = self._decode_forced(curr_cs, cur_enc_states)
            true_score = F.log_softmax(score, dim=2).gather(2, curr_cs.unsqueeze(2))
            nonzero = curr_cs.ne(0).float()
            scores = (true_score.squeeze(2) * nonzero).sum(1)
            seqlens = nonzero.sum(1)
            scores /= seqlens
            cand_scores.append(scores)

        max_len = max(len(c) for c in cand_scores)
        cand_scores = torch.cat(
            [pad(c, max_len, pad=self.NULL_IDX).unsqueeze(0) for c in cand_scores], 0
        )
        return cand_scores

    def forward(
        self, xs, ys=None, cands=None, prev_enc=None, maxlen=None, seq_len=None
    ):
        """
        Get output predictions from the model.

        :param xs:          (bsz x seqlen) LongTensor input to the encoder
        :param ys:          expected output from the decoder. used for teacher
                            forcing to calculate loss.
        :param cands:       set of candidates to rank
        :param prev_enc:    if you know you'll pass in the same xs multiple
                            times, you can pass in the encoder output from the
                            last forward pass to skip recalcuating the same
                            encoder output.
        :param maxlen:      max number of tokens to decode. if not set, will
                            use the length of the longest label this model
                            has seen. ignored when ys is not None.
        :param seq_len      this is the sequence length of the input (xs), i.e.
                            xs.size(1). we use this to recover the proper
                            output sizes in the case when we distribute over
                            multiple gpus

        :returns: scores, candidate scores, and encoder states
            scores contains the model's predicted token scores.
                (bsz x seqlen x num_features)
            candidate scores are the score the model assigned to each candidate
                (bsz x num_cands)
            encoder states are the (output, hidden, attn_mask) states from the
                encoder. feed this back in to skip encoding on the next call.
        """
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_states = self._encode(xs, prev_enc)

        # rank candidates if they are available
        cand_scores = None
        if cands is not None:
            cand_inds = [i for i in range(cands.size(0))]
            cand_scores = self._rank(cands, cand_inds, encoder_states)

        if ys is not None:
            # use teacher forcing
            scores = self._decode_forced(ys, encoder_states)
        else:
            scores = self._decode(encoder_states, maxlen or self.longest_label)

        if seq_len is not None:
            # when using multiple gpus, we need to make sure output of
            # encoder is correct size for gathering; we recover this with
            # the parameter seq_len
            if encoder_states[0].size(1) < seq_len:
                out_pad_tensor = torch.zeros(
                    encoder_states[0].size(0),
                    seq_len - encoder_states[0].size(1),
                    encoder_states[0].size(2),
                ).cuda()
                new_out = torch.cat([encoder_states[0], out_pad_tensor], 1)
                encoder_states = (new_out, encoder_states[1], encoder_states[2])

        return scores, cand_scores, encoder_states


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
            x_lens = torch.sum(attn_mask.int(), dim=1)
            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output, batch_first=True)
        if self.dirs > 1:
            # project to decoder dimension by taking sum of forward and back
            if isinstance(self.rnn, nn.LSTM):
                hidden = (
                    hidden[0].view(-1, self.dirs, bsz, self.hsz).sum(1),
                    hidden[1].view(-1, self.dirs, bsz, self.hsz).sum(1),
                )
            else:
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).sum(1)

        return encoder_output, hidden, attn_mask


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

    def forward(self, xs, hidden=None, attn_params=None):
        """
        Decode from input tokens.

        :param xs:          (bsz x seqlen) LongTensor of input token indices
        :param hidden:      hidden state to feed into decoder. default (None)
                            initializes tensors using the RNN's defaults.
        :param attn_params: (optional) tuple containing attention parameters,
                            default AttentionLayer needs encoder_output states
                            and attention mask (e.g. encoder_input.ne(0))

        :returns:           output state(s), hidden state.
                            output state of the encoder. for an RNN, this is
                            (bsz, seq_len, num_directions * hiddensize).
                            hidden state will be same dimensions as input
                            hidden state. for an RNN, this is a tensor of sizes
                            (bsz, numlayers * num_directions, hiddensize).
        """
        # sequence indices => sequence embeddings
        xes = self.dropout(self.lt(xs))

        if self.attn_time == 'pre':
            # modify input vectors with attention
            xes, _attw = self.attention(xes, hidden, attn_params)

        # feed tokens into rnn
        output, new_hidden = self.rnn(xes, hidden)

        if self.attn_time == 'post':
            # modify output vectors with attention
            output, _attw = self.attention(output, new_hidden, attn_params)

        return output, new_hidden


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

        self.padding_idx = padding_idx if shared_weight is not None else -1

        # embedding to scores
        if shared_weight is None:
            # just a regular linear layer
            self.e2s = nn.Linear(embeddingsize, num_features, bias=True)
        else:
            # use shared weights and a bias layer instead
            if padding_idx == 0:
                num_features -= 1  # don't include padding
                shared_weight = shared_weight.narrow(0, 1, num_features)
            elif padding_idx > 0:
                raise RuntimeError('nonzero pad_idx not yet implemented')
            self.weight = Parameter(shared_weight)
            self.bias = Parameter(torch.Tensor(num_features))
            self.reset_parameters()
            self.e2s = lambda x: F.linear(x, self.weight, self.bias)

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
                self.o2e = lambda x: x

    def reset_parameters(self):
        """
        Reset bias param.
        """
        if hasattr(self, 'bias'):
            stdv = 1.0 / math.sqrt(self.bias.size(0))
            self.bias.data.uniform_(-stdv, stdv)

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
            logit = self.e2s(active.view(-1, self.esz))

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
            scores = self.e2s(e)

        if self.padding_idx == 0:
            pad_score = scores.new(scores.size(0), scores.size(1), 1).fill_(-NEAR_INF)
            scores = torch.cat([pad_score, scores], dim=-1)

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
                attn_w_premask.masked_fill_((1 - attn_mask), -NEAR_INF)
            attn_weights = F.softmax(attn_w_premask, dim=1)

        # apply the attention weights to the encoder states
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        # concatenate the input and encoder states
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        # combine them with a linear layer and tanh activation
        output = torch.tanh(self.attn_combine(merged).unsqueeze(1))

        return output, attn_weights
