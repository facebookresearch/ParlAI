# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn.functional as F


class Seq2seq(nn.Module):
    RNN_OPTS = {'rnn': nn.RNN, 'gru': nn.GRU, 'lstm': nn.LSTM}

    def __init__(self, opt, num_features,
                 padding_idx=0, start_idx=1, end_idx=2, longest_label=1):
        super().__init__()
        self.opt = opt

        self.rank = opt['rank_candidates']
        self.attn_type = opt['attention']

        self.NULL_IDX = padding_idx
        self.END_IDX = end_idx
        self.register_buffer('START', torch.LongTensor([start_idx]))
        self.longest_label = longest_label

        rnn_class = Seq2seq.RNN_OPTS[opt['rnn_class']]
        self.decoder = Decoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            share_output=opt['lookuptable'] in ['dec_out', 'all'],
            attn_type=opt['attention'], attn_length=opt['attention_length'],
            attn_time=opt.get('attention_time'),
            bidir_input=opt['bidirectional'])

        shared_lt = (self.decoder.lt
                     if opt['lookuptable'] in ['enc_dec', 'all'] else None)
        shared_rnn = self.decoder.rnn if opt['decoder'] == 'shared' else None
        self.encoder = Encoder(
            num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
            emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
            num_layers=opt['numlayers'], dropout=opt['dropout'],
            bidirectional=opt['bidirectional'],
            shared_lt=shared_lt, shared_rnn=shared_rnn)

        if self.rank:
            self.ranker = Ranker(self.decoder, padding_idx=self.NULL_IDX,
                                 attn_type=opt['attention'])

    def forward(self, xs, ys=None, cands=None, valid_cands=None):
        bsz = len(xs)
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        enc_out, hidden = self.encoder(xs)
        attn_mask = xs.ne(0).float() if self.attn_type != 'none' else None
        start = Variable(self.START, requires_grad=False)
        starts = start.expand(bsz, 1)

        predictions = []
        scores = []
        text_cand_inds = None
        if ys is not None:
            y_in = ys.narrow(1, 0, ys.size(1) - 1)
            xs = torch.cat([starts, y_in], 1)
            if self.attn_type == 'none':
                preds, score, _h = self.decoder(xs, hidden, enc_out, attn_mask)
                predictions.append(preds)
                scores.append(score)
            else:
                for i in range(ys.size(1)):
                    xi = xs.select(1, i)
                    preds, score, hidden = self.decoder(xi, hidden, enc_out, attn_mask)
                    predictions.append(preds)
                    scores.append(score)
        else:
            # just predict
            done = [False for _ in range(bsz)]
            total_done = 0
            xs = starts

            for _ in range(self.longest_label):
                # generate at most longest_label tokens
                preds, score, hidden = self.decoder(xs, hidden, enc_out, attn_mask)
                scores.append(score)
                xs = preds
                predictions.append(preds)

                # check if we've produced the end token
                for b in range(bsz):
                    if not done[b]:
                        # only add more tokens for examples that aren't done
                        if preds.data[b][0] == self.END_IDX:
                            # if we produced END, we're done
                            done[b] = True
                            total_done += 1
                if total_done == bsz:
                    # no need to generate any more
                    break
            if self.rank and cands is not None:
                text_cand_inds = self.ranker(cands, valid_cands, start,
                                             hidden, enc_out, attn_mask)

        if predictions:
            predictions = torch.cat(predictions, 1)
        if scores:
            scores = torch.cat(scores, 1)
        return predictions, scores, text_cand_inds


class Encoder(nn.Module):
    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1,
                 bidirectional=False, shared_lt=None, shared_rnn=None,
                 sparse=False):
        super().__init__()

        self.dropout = dropout
        self.layers = num_layers
        self.dirs = 2 if bidirectional else 1
        self.hsz = hidden_size

        # we put zeros in here
        self.buffers = {}

        if shared_lt is None:
            self.lt = nn.Embedding(num_features, emb_size,
                                   padding_idx=padding_idx,
                                   sparse=sparse)
        else:
            self.lt = shared_lt

        if shared_rnn is None:
            self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                                 dropout=dropout, batch_first=True,
                                 bidirectional=bidirectional)
        elif bidirectional:
            raise RuntimeError('Cannot share decoder with bidir encoder.')
        else:
            self.rnn = shared_rnn

    def zeros(self, typeof):
        cur_type = typeof.data.type()
        if cur_type not in self.buffers:
            self.buffers[cur_type] = typeof.data.new(
                self.layers * self.dirs, 1, self.hsz).float().fill_(0)
        return self.buffers[cur_type]

    def forward(self, xs):
        bsz = len(xs)

        # embed input tokens
        xes = F.dropout(self.lt(xs), p=self.dropout, training=self.training)
        x_lens = [x for x in torch.sum((xs > 0).int(), dim=1).data]
        xes_packed = pack_padded_sequence(xes, x_lens, batch_first=True)

        zeros = self.zeros(xs)
        if zeros.size(1) != bsz:
            zeros.resize_(self.layers * self.dirs, bsz, self.hsz).fill_(0)
        h0 = Variable(zeros, requires_grad=False)

        if type(self.rnn) == nn.LSTM:
            encoder_output_packed, hidden = self.rnn(xes_packed, (h0, h0))
            if self.dirs > 1:
                # take elementwise max between forward and backward hidden states
                hidden = (hidden[0].view(-1, self.dirs, bsz, self.hsz).max(1)[0],
                          hidden[1].view(-1, self.dirs, bsz, self.hsz).max(1)[0])
        else:
            encoder_output_packed, hidden = self.rnn(xes_packed, h0)

            if self.dirs > 1:
                # take elementwise max between forward and backward hidden states
                hidden = hidden.view(-1, self.dirs, bsz, self.hsz).max(1)[0]
        encoder_output, _ = pad_packed_sequence(encoder_output_packed,
                                                batch_first=True)
        return encoder_output, hidden


class Decoder(nn.Module):
    def __init__(self, num_features, padding_idx=0, rnn_class='lstm',
                 emb_size=128, hidden_size=128, num_layers=2, dropout=0.1,
                 bidir_input=False, share_output=True,
                 attn_type='none', attn_length=-1, attn_time='pre',
                 sparse=False):
        super().__init__()

        if padding_idx != 0:
            raise RuntimeError('This module\'s output layer needs to be fixed '
                               'if you want a padding_idx other than zero.')

        self.dropout = dropout
        self.layers = num_layers
        self.hsz = hidden_size

        self.lt = nn.Embedding(num_features, emb_size, padding_idx=padding_idx,
                               sparse=sparse)
        self.rnn = rnn_class(emb_size, hidden_size, num_layers,
                             dropout=dropout, batch_first=True)

        # rnn output to embedding
        if hidden_size != emb_size:
            self.o2e = RandomProjection(hidden_size, emb_size)
            # other option here is to learn these weights
            # self.o2e = nn.Linear(hidden_size, emb_size, bias=False)
        else:
            # no need for any transformation here
            self.o2e = lambda x: x
        # embedding to scores, use custom linear to possibly share weights
        shared_weight = self.lt.weight if share_output else None
        self.e2s = Linear(emb_size, num_features, bias=False,
                          shared_weight=shared_weight)
        self.shared = shared_weight is not None

        self.attn_type = attn_type
        self.attn_time = attn_time
        self.attention = AttentionLayer(attn_type=attn_type,
                                        hidden_size=hidden_size,
                                        emb_size=emb_size,
                                        bidirectional=bidir_input,
                                        attn_length=attn_length,
                                        attn_time=attn_time)

    def forward(self, xs, hidden, encoder_output, attn_mask=None):
        xes = F.dropout(self.lt(xs), p=self.dropout, training=self.training)
        if self.attn_time == 'pre':
            xes = self.attention(xes, hidden, encoder_output, attn_mask)
        if xes.dim() == 2:
            # if only one token inputted, sometimes needs unsquezing
            xes.unsqueeze_(1)
        output, new_hidden = self.rnn(xes, hidden)
        if self.attn_time == 'post':
            output = self.attention(output, new_hidden, encoder_output, attn_mask)

        e = self.o2e(output)
        scores = F.dropout(self.e2s(e), p=self.dropout, training=self.training)
        # select top scoring index, excluding the padding symbol (at idx zero)
        _max_score, idx = scores.narrow(2, 1, scores.size(2) - 1).max(2)
        preds = idx.add_(1)

        return preds, scores, new_hidden


class Ranker(nn.Module):
    def __init__(self, decoder, padding_idx=0, attn_type='none'):
        super().__init__()
        self.decoder = decoder
        # put intermediate states in these tensors
        self.buffers = {}
        self.NULL_IDX = padding_idx
        self.attn_type = attn_type

    def buffer(self, typeof, name, sz):
        key = name + '_' + typeof.data.type()
        if key not in self.buffers:
            self.buffers[key] = typeof.data.new(sz)
        return self.buffers[key].resize_(sz).fill_(0)

    def forward(self, cands, cand_inds, start, hidden, enc_out, attn_mask):
        cell = None
        if type(hidden) == tuple:
            # for lstms, split hidden state into parts
            hidden, cell = hidden

        num_hid = hidden.size(0)
        bsz = hidden.size(1)
        esz = hidden.size(2)
        cands_per_ex = cands.size(1)
        words_per_cand = cands.size(2)

        # score each candidate separately
        # cands are exs_with_cands x cands_per_ex x words_per_cand
        # cview is total_cands x words_per_cand
        cview = cands.view(-1, words_per_cand)
        total_cands = cview.size(0)
        starts = start.expand(total_cands).unsqueeze(1)

        if len(cand_inds) != hidden.size(1):
            # select hidden states which have associated cands
            cand_indices = Variable(start.data.new([i[0] for i in cand_inds]))
            hidden = hidden.index_select(1, cand_indices)

        h_exp = (
            # expand hidden states so each cand has an initial hidden state
            # cands for the same input have the same initial hidden state
            hidden.unsqueeze(2)
            .expand(num_hid, bsz, cands_per_ex, esz)
            .contiguous()
            .view(num_hid, -1, esz))

        if cell is None:
            cands_hn = h_exp
        if cell is not None:
            if len(cand_inds) != cell.size(1):
                # only use cell state from inputs with associated candidates
                cell = cell.index_select(1, cand_indices)
            c_exp = (
                cell.unsqueeze(2)
                .expand(num_hid, bsz, cands_per_ex, esz)
                .contiguous()
                .view(num_hid, -1, esz))
            cands_hn = (h_exp, c_exp)

        cand_scores = Variable(self.buffer(hidden, 'cand_scores', total_cands))
        cand_lens = Variable(self.buffer(start, 'cand_lens', total_cands))

        if self.attn_type == 'none':
            # process entire sequence at once
            if cview.size(1) > 1:
                # feed in START + cands[:-2]
                cands_in = cview.narrow(1, 0, cview.size(1) - 1)
                starts = torch.cat([starts, cands_in], 1)
            _preds, score, _h = self.decoder(starts, cands_hn, enc_out, attn_mask)

            for i in range(cview.size(1)):
                # calculate score at each token
                cs = cview.select(1, i)
                non_nulls = cs.ne(self.NULL_IDX)
                cand_lens += non_nulls.long()
                score_per_cand = torch.gather(score.select(1, i), 1,
                                              cs.unsqueeze(1))
                cand_scores += score_per_cand.squeeze() * non_nulls.float()
        else:
            # using attention
            if len(cand_inds) != len(enc_out):
                # select only encoder output matching xs we want
                indices = Variable(start.data.new([i[0] for i in cand_inds]))
                enc_out = enc_out.index_select(0, indices)
                attn_mask = attn_mask.index_select(0, indices)

            seq_len = enc_out.size(1)
            cands_enc_out = (
                enc_out.unsqueeze(1)
                .expand(bsz, cands_per_ex, seq_len, esz)
                .contiguous()
                .view(-1, seq_len, esz)
            )
            cands_attn_mask = (
                attn_mask.unsqueeze(1)
                .expand(bsz, cands_per_ex, seq_len)
                .contiguous()
                .view(-1, seq_len)
            )

            cs = starts
            for i in range(cview.size(1)):
                # process one token at a time
                _preds, score, _h = self.decoder(cs, cands_hn, cands_enc_out,
                                             cands_attn_mask)
                cs = cview.select(1, i)
                non_nulls = cs.ne(self.NULL_IDX)
                cand_lens += non_nulls.long()
                score_per_cand = torch.gather(score.squeeze(), 1,
                                              cs.unsqueeze(1))
                cand_scores += score_per_cand.squeeze() * non_nulls.float()

        # set empty scores to -1, so when divided by 0 they become -inf
        cand_scores -= cand_lens.eq(0).float()
        # average the scores per token
        cand_scores /= cand_lens.float()

        cand_scores = cand_scores.view(cands.size(0), cands.size(1))
        _srtd_scores, text_cand_inds = cand_scores.sort(1, True)

        return text_cand_inds


class Linear(nn.Module):
    """Custom Linear layer which allows for sharing weights (e.g. with an
    nn.Embedding layer).
    """
    def __init__(self, in_features, out_features, bias=True,
                 shared_weight=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.shared = shared_weight is not None

        # init weight
        if not self.shared:
            self.weight = Parameter(torch.Tensor(out_features, in_features))
        else:
            if (shared_weight.size(0) != out_features or
                    shared_weight.size(1) != in_features):
                raise RuntimeError('wrong dimensions for shared weights')
            self.weight = shared_weight

        # init bias
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        if not self.shared:
            # weight is shared so don't overwrite it
            self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        weight = self.weight
        if self.shared:
            # detach weight to prevent gradients from changing weight
            # (but need to detach every time so weights are up to date)
            weight = weight.detach()
        return F.linear(input, weight, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class RandomProjection(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features),
                                requires_grad=False)  # fix weights
        self.reset_parameters()

    def reset_parameters(self):
        # experimentally: std=1 appears to affect scale too much
        self.weight.data.normal_(std=0.1)
        # other init option: set randomly to 1 or -1
        # self.weight.data.bernoulli_(self.weight.fill_(0.5)).mul_(2).sub_(1)

    def forward(self, input):
        return F.linear(input, self.weight)


class AttentionLayer(nn.Module):
    def __init__(self, attn_type, hidden_size, emb_size, bidirectional=False,
                 attn_length=-1, attn_time='pre'):
        super().__init__()
        self.attention = attn_type

        if self.attention != 'none':
            hsz = hidden_size
            num_dirs = 2 if bidirectional else 1
            hszXdirs = hsz * num_dirs
            if attn_time == 'pre':
                # attention happens on the input embeddings
                input_dim = emb_size
            elif attn_time == 'post':
                # attention happens on the output of the rnn
                input_dim = hsz
            else:
                raise RuntimeError('unsupported attention time')
            self.attn_combine = nn.Linear(hszXdirs + input_dim, input_dim,
                                          bias=False)

            if self.attention == 'local':
                # local attention over fixed set of output states
                if attn_length < 0:
                    raise RuntimeError('Set attention length to > 0.')
                self.max_length = attn_length
                # combines input and previous hidden output layer
                self.attn = nn.Linear(hsz + emb_size, attn_length, bias=False)
                # combines attention weights with encoder outputs
            elif self.attention == 'concat':
                self.attn = nn.Linear(hsz + hszXdirs, hsz, bias=False)
                self.attn_v = nn.Linear(hsz, 1, bias=False)
            elif self.attention == 'general':
                # equivalent to dot if attn is identity
                self.attn = nn.Linear(hsz, hszXdirs, bias=False)

    def forward(self, xes, hidden, enc_out, attn_mask=None):
        if self.attention == 'none':
            return xes

        if type(hidden) == tuple:
            # for lstms use the "hidden" state not the cell state
            hidden = hidden[0]
        last_hidden = hidden[-1]  # select hidden state from last RNN layer

        if self.attention == 'local':
            if enc_out.size(1) > self.max_length:
                offset = enc_out.size(1) - self.max_length
                enc_out = enc_out.narrow(1, offset, self.max_length)
            h_merged = torch.cat((xes.squeeze(1), last_hidden), 1)
            attn_weights = F.softmax(self.attn(h_merged), dim=1)
            if attn_weights.size(1) > enc_out.size(1):
                attn_weights = attn_weights.narrow(1, 0, enc_out.size(1))
        else:
            hid = last_hidden.unsqueeze(1)
            if self.attention == 'concat':
                hid = hid.expand(last_hidden.size(0),
                                 enc_out.size(1),
                                 last_hidden.size(1))
                h_merged = torch.cat((enc_out, hid), 2)
                active = F.tanh(self.attn(h_merged))
                attn_w_premask = self.attn_v(active).squeeze(2)
            elif self.attention == 'dot':
                if hid.size(2) != enc_out.size(2):
                    # enc_out has two directions, so double hid
                    hid = torch.cat([hid, hid], 2)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            elif self.attention == 'general':
                hid = self.attn(hid)
                attn_w_premask = (
                    torch.bmm(hid, enc_out.transpose(1, 2)).squeeze(1))
            # calculate activation scores
            if attn_mask is not None:
                # remove activation from NULL symbols
                attn_w_premask -= (1 - attn_mask) * 1e20
            attn_weights = F.softmax(attn_w_premask, dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), enc_out)
        merged = torch.cat((xes.squeeze(1), attn_applied.squeeze(1)), 1)
        output = F.tanh(self.attn_combine(merged).unsqueeze(1))

        return output
