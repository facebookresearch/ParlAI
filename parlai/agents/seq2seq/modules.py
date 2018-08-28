# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F
from parlai.core.torch_agent import Beam
from parlai.core.dict import DictionaryAgent
import os


def pad(tensor, length, dim=0):
    if tensor.size(dim) < length:
        return torch.cat(
            [tensor, tensor.new(*tensor.size()[:dim],
                                length - tensor.size(dim),
                                *tensor.size()[dim + 1:]).zero_()],
            dim=dim)
    else:
        return tensor


class Seq2seq(nn.Module):
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, opt, num_features,
                 padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()
        self.opt = opt

        self.attn_type = opt['attention']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Seq2seq.RNN_OPTS[opt['rnn_class']]
        self.decoder = RNNDecoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            share_output=opt['lookuptable'] in ['dec_out', 'all'],
            attn_type=opt['attention'], attn_length=opt['attention_length'],
            attn_time=opt.get('attention_time'),
            bidir_input=opt['bidirectional'],
            numsoftmax=opt.get('numsoftmax', 1))

        shared_lt = (self.decoder.lt
                     if opt['lookuptable'] in ['enc_dec', 'all'] else None)
        shared_rnn = self.decoder.rnn if opt['decoder'] == 'shared' else None
        self.encoder = RNNEncoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            bidirectional=opt['bidirectional'],
            shared_lt=shared_lt, shared_rnn=shared_rnn)

        if opt['rank_candidates']:
            self.ranker = Ranker(
                self.decoder,
                self.START,
                padding_idx=self.NULL_IDX,
                attn_type=opt['attention'])

        self.beam_log_freq = opt.get('beam_log_freq', 0.0)
        if self.beam_log_freq > 0.0:
            self.dict = DictionaryAgent(opt)
            self.beam_dump_filecnt = 0
            self.beam_dump_path = opt['model_file'] + '.beam_dump'
            if not os.path.exists(self.beam_dump_path):
                os.makedirs(self.beam_dump_path)

    def _encode(self, xs, x_lens=None, prev_enc=None):
        if prev_enc is not None:
            return prev_enc
        else:
            enc_out, hidden = self.encoder(xs, x_lens)
            attn_mask = xs.ne(0).float() if self.attn_type != 'none' else None
            return hidden, enc_out, attn_mask

    def _rank(self, cand_params, encoder_states):
        if cand_params is not None:
            return self.ranker.forward(cand_params, encoder_states)
        return None

    def _starts(self, bsz):
        return self.START.detach().expand(bsz, 1)

    def _decode_forced(self, ys, encoder_states):
        bsz = ys.size(0)
        seqlen = ys.size(1)

        hidden = encoder_states[0]
        attn_params = (encoder_states[1], encoder_states[2])

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
                xi = xs.select(1, i)
                output, hidden = self.decoder(xi, hidden, attn_params)
                score = self.output(output)
                scores.append(score)

        return scores

    def _decode(self, encoder_states, maxlen):
        hidden = encoder_states[0]
        attn_params = (encoder_states[1], encoder_states[2])
        bsz = hidden.size(0)

        xs = self._starts(bsz)  # input start token

        scores = []
        for _ in range(maxlen):
            # generate at most longest_label tokens
            output, hidden = self.decoder(xs, hidden, attn_params)
            score = self.output(output)
            scores.append(score)
            xs = scores.max(1)[0]  # next input is current predicted output

        return scores

    def forward(self, xs, x_lens=None, ys=None, cand_params=None,
                prev_enc=None, rank_during_training=False, maxlen=None):
        """Get output predictions from the model.

        :param xs: (bsz x seqlen) LongTensor input to the encoder
        :param x_lens: (list of ints) length of each input sequence
        :param ys: expected output from the decoder. used for teacher forcing to calculate loss.
        :param cand_params: set of candidates to rank, and indices to match candidates with their appropriate xs
        :param prev_enc: if you know you'll pass in the same xs multiple times, you can pass in the encoder output from the last forward pass to skip recalcuating the same encoder output
        :
        """
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        encoder_states = self.encode(xs, x_lens, prev_enc)

        # rank candidates if they are available
        cand_scores = self._rank(cand_params, encoder_states)

        if ys is not None:
            # use teacher forcing
            scores = self._decode_forced(ys, encoder_states)
        else:
            scores = self._decode(encoder_states, maxlen or self.longest_label)

        if isinstance(scores, list):
            scores = torch.cat(scores, 1)

        return scores, cand_scores, encoder_states


class RNNEncoder(nn.Module):
    """RNN Encoder."""

    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1,
                 bidirectional=False, shared_lt=None, shared_rnn=None,
                 sparse=False):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hidden_size

        if shared_lt is None:
            self.lt = nn.Embedding(num_features, emb_size,
                                   padding_idx=padding_idx,
                                   sparse=sparse)
        else:
            self.lt = shared_lt

        if shared_rnn is None:
            self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                                 dropout=dropout if num_layers > 1 else 0,
                                 batch_first=True, bidirectional=bidirectional)
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def forward(self, xs, x_lens=None):
        """Encode sequence.

        :param xs: (bsz x seqlen) LongTensor of input token indices
        """
        bsz = len(xs)

        # embed input tokens
        xes = self.dropout(self.lt(xs))
        try:
            if x_lens is None:
                x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
            xes = pack_padded_sequence(xes, x_lens, batch_first=True)
            packed = True
        except ValueError:
            # packing failed, don't pack then
            packed = False

        encoder_output, hidden = self.rnn(xes)
        if packed:
            encoder_output, _ = pad_packed_sequence(encoder_output,
                                                    batch_first=True)
        if self.dirs > 1:
            # project to decoder dimension by taking sum of forward and back
            if isinstance(self.rnn, nn.LSTM):
                hidden = (hidden[0].view(-1, self.dirs, bsz, self.hsz).sum(1),
                          hidden[1].view(-1, self.dirs, bsz, self.hsz).sum(1))
            else:
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).sum(1)

        return encoder_output, hidden


class RNNDecoder(nn.Module):
    """Recurrent decoder module.

    Can be used as a standalone language model or paired with an encoder.
    """

    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1,
                 bidir_input=False, attn_type='none', attn_time='pre',
                 attn_length=-1, sparse=False):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.layers = num_layers
        self.hsz = hidden_size
        self.esz = emb_size

        self.lt = nn.Embedding(num_features, emb_size, padding_idx=padding_idx,
                               sparse=sparse)
        self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                             dropout=dropout if num_layers > 1 else 0,
                             batch_first=True)

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(attn_type=attn_type,
                                        hidden_size=hidden_size,
                                        emb_size=emb_size,
                                        bidirectional=bidir_input,
                                        attn_length=attn_length,
                                        attn_time=attn_time)

    def forward(self, xs, hidden=None, attn_params=None):
        """Decode from input tokens.

        :param xs:          (bsz x seqlen) LongTensor of input token indices
        :param hidden:      hidden state to feed into decoder. default (None)
                            initializes tensors using the RNN's defaults.
        :param attn_params: (optional) tuple containing attention parameters,
                            default AttentionLayer needs encoder_output states
                            and attention mask (e.g. encoder_input.ne(0))

        :returns:           output state(s), hidden state.
                            output state of the encoder. for an RNN, this is
                            (bsz, seq_len, num_directions * hidden_size).
                            hidden state will be same dimensions as input
                            hidden state. for an RNN, this is a tensor of sizes
                            (bsz, num_layers * num_directions, hidden_size).
        """
        # sequence indices => sequence embeddings
        xes = self.dropout(self.lt(xs))

        if self.attn_time == 'pre':
            # modify input vectors with attention
            xes = self.attention(xes, hidden, attn_params)

        # feed tokens into rnn
        output, new_hidden = self.rnn(xes, hidden)

        if self.attn_time == 'post':
            # modify output vectors with attention
            output = self.attention(output, new_hidden, attn_params)

        return output, new_hidden


class OutputLayer(nn.Module):
    """Takes in final states and returns distribution over candidates."""

    def __init__(self, num_features, hidden_size, emb_size, numsoftmax=1,
                 shared_weight=None):
        """Initialize output layer.

        :param num_features:  number of candidates to rank
        :param hidden_size:   (last) dimension of the input vectors
        :param emb_size:      (last) dimension of the candidate vectors
        :param num_softmax:   (default 1) number of softmaxes to calculate.
                              see arxiv.org/abs/1711.03953 for more info.
                              increasing this slows down computation but can
                              add more expressivity to the embeddings.
        :param shared_weight: (num_features x esz) vector of
        """
        # embedding to scores
        if shared_weight is None:
            # just a regular linear layer
            self.e2s = nn.Linear(emb_size, num_features, bias=True)
        else:
            # use shared weights and a bias layer instead
            self.weight = shared_weight
            self.bias = Parameter(torch.Tensor(num_features))
            self.reset_parameters()
            self.e2s = lambda x: F.linear(x, self.weight, self.bias)

        self.numsoftmax = numsoftmax
        if numsoftmax > 1:
            self.softmax = nn.Softmax(dim=1)
            self.prior = nn.Linear(hidden_size, numsoftmax, bias=False)
            self.latent = nn.Linear(hidden_size, numsoftmax * emb_size)
            self.activation = nn.Tanh()
        else:
            # rnn output to embedding
            if hidden_size != emb_size:
                # learn projection to correct dimensions
                self.o2e = nn.Linear(hidden_size, emb_size, bias=True)
            else:
                # no need for any transformation here
                self.o2e = lambda x: x

    def reset_parameters(self):
        """Reset bias param."""
        if hasattr(self, 'bias'):
            stdv = 1. / math.sqrt(self.bias.size(1))
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        """Compute scores from inputs.

        :param input: (bsz x seq_len x num_directions * hidden_size) tensor of
                       states, e.g. the output states of an RNN

        :returns: (bsz x seqlen x num_cands) scores for each candidate
        """
        # next compute scores over dictionary
        if self.numsoftmax > 1:
            bsz = input.size(0)
            seqlen = input.size(1) if input.dim() > 1 else 1

            # first compute different softmax scores based on input vec
            # hsz => num_softmax * esz
            latent = self.latent(input)
            active = self.dropout(self.activation(latent))
            # esz => num_features
            logit = self.e2s(active.view(-1, self.esz))

            # calculate priors: distribution over which softmax scores to use
            # hsz => num_softmax
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

        return scores


class Ranker(object):
    def __init__(self, decoder, start_token, padding_idx=0, attn_type='none'):
        super().__init__()
        self.decoder = decoder
        self.START = start_token
        self.NULL_IDX = padding_idx
        self.attn_type = attn_type

    def _starts(self, bsz):
        return self.START.detach().expand(bsz, 1)

    def forward(self, cand_params, encoder_states):
        cands, cand_inds = cand_params
        hidden, enc_out, attn_mask = encoder_states

        hid, cell = (hidden, None) if isinstance(hidden, torch.Tensor) else hidden
        if len(cand_inds) != hid.size(1):
            cand_indices = self.START.detach().new(cand_inds)
            hid = hid.index_select(1, cand_indices)
            if cell is None:
                hidden = hid
            else:
                cell = cell.index_select(1, cand_indices)
                hidden = (hid, cell)
            enc_out = enc_out.index_select(0, cand_indices)
            if attn_mask is not None:
                attn_mask = attn_mask.index_select(0, cand_indices)

        cand_scores = []

        for i in range(len(cands)):
            curr_cs = cands[i]

            n_cs = curr_cs.size(0)
            starts = self._starts(n_cs).unsqueeze(1)
            scores = 0
            seqlens = 0
            # select just the one hidden state
            if isinstance(hidden, torch.Tensor):
                nl = hidden.size(0)
                hsz = hidden.size(-1)
                cur_hid = hidden.select(1, i).unsqueeze(1).expand(nl, n_cs, hsz)
            else:
                nl = hidden[0].size(0)
                hsz = hidden[0].size(-1)
                cur_hid = (hidden[0].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous(),
                           hidden[1].select(1, i).unsqueeze(1).expand(nl, n_cs, hsz).contiguous())

            cur_enc, cur_mask = None, None
            if attn_mask is not None:
                cur_mask = attn_mask[i].unsqueeze(0).expand(n_cs, attn_mask.size(-1))
                cur_enc = enc_out[i].unsqueeze(0).expand(n_cs, enc_out.size(1), hsz)
            # this is pretty much copied from the training forward above
            if curr_cs.size(1) > 1:
                c_in = curr_cs.narrow(1, 0, curr_cs.size(1) - 1)
                xs = torch.cat([starts, c_in], 1)
            else:
                xs, c_in = starts, curr_cs
            if self.attn_type == 'none':
                preds, score, cur_hid = self.decoder(xs, cur_hid, cur_enc, cur_mask)
                true_score = F.log_softmax(score, dim=2).gather(
                    2, curr_cs.unsqueeze(2))
                nonzero = curr_cs.ne(0).float()
                scores = (true_score.squeeze(2) * nonzero).sum(1)
                seqlens = nonzero.sum(1)
            else:
                for i in range(curr_cs.size(1)):
                    xi = xs.select(1, i)
                    ci = curr_cs.select(1, i)
                    preds, score, cur_hid = self.decoder(xi, cur_hid, cur_enc, cur_mask)
                    true_score = F.log_softmax(score, dim=2).gather(
                        2, ci.unsqueeze(1).unsqueeze(2))
                    nonzero = ci.ne(0).float()
                    scores += true_score.squeeze(2).squeeze(1) * nonzero
                    seqlens += nonzero

            scores /= seqlens  # **len_penalty?
            cand_scores.append(scores)

        max_len = max(len(c) for c in cand_scores)
        cand_scores = torch.cat([pad(c, max_len).unsqueeze(0) for c in cand_scores], 0)
        return cand_scores


class AttentionLayer(nn.Module):
    """Computes attention between hidden and encoder states.

    See arxiv.org/abs/1508.04025 for more info on each attention type.
    """

    def __init__(self, attn_type, hidden_size, emb_size, bidirectional=False,
                 attn_length=-1, attn_time='pre'):
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hidden_size
            hszXdirs = hsz * (2 if bidirectional else 1)
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = emb_size
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')

            # linear layer for combining applied attention weights with input
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

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
        """Compute attention over attn_params given input and hidden states.

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
            return xes

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
                import pdb; pdb.set_trace()  # is this the best operation?
                attn_w_premask -= (1 - attn_mask) * 1e20
            attn_weights = F.softmax(attn_w_premask, dim=1)

        # apply the attention weights to the encoder states
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        # concatenate the input and encoder states
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        # combine them with a linear layer and tanh activation
        output = F.tanh(self.attn_combine(merged).unsqueeze(1))
        print('make sure last_hidden == xes')
        import pdb; pdb.set_trace()
        # TODO: remove tanh?

        return output, attn_weights
