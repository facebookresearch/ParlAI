# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.

from parlai.agents.seq2seq.modules import Seq2seq, Encoder, Decoder, AttentionLayer

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class PersonaSeq2seq(Seq2seq):
    """Same as normal Seq2seq but does attention over personas."""
    def __init__(self, opt, num_features, *args, **kwargs):
        super().__init__(opt, num_features, *args, **kwargs)
        rnn_class = Seq2seq.RNN_OPTS[opt['rnn_class']]

        self.persona_attention = opt['persona_attention']
        self.persona_encoding = opt['persona_encoding']

        # override attention layer for persona attention
        if self.persona_attention == 'context':
            # do nothign
            pass
        elif self.persona_attention == 'persona':

            if self.persona_encoding == 'bow':
                self.hszXdirs = opt['hiddensize']
                self.persona_encoder = nn.EmbeddingBag(num_features,
                    opt['embeddingsize'], scale_grad_by_freq=True)
                self.bow2hsz = nn.Linear(opt['embeddingsize'], opt['hiddensize'])
                self.decoder.attention = AttentionLayer(
                    attn_type=opt['attention'],
                    hidden_size=opt['hiddensize'],
                    emb_size=opt['embeddingsize'],
                    bidirectional=False,
                    attn_length=opt['attention_length'],
                    attn_time=opt['attention_time'])
            else:
                self.hszXdirs = opt['hiddensize'] * (2 if opt['persona_bidirectional'] else 1)
                self.decoder.attention = AttentionLayer(
                    attn_type=opt['attention'],
                    hidden_size=opt['hiddensize'],
                    emb_size=opt['embeddingsize'],
                    bidirectional=opt['persona_bidirectional'],
                    attn_length=opt['attention_length'],
                    attn_time=opt['attention_time'])

                # separate encoder for personas
                self.persona_encoder = Encoder(
                    num_features, padding_idx=self.NULL_IDX, rnn_class=rnn_class,
                    emb_size=opt['embeddingsize'], hidden_size=opt['hiddensize'],
                    num_layers=opt['persona_numlayers'], dropout=opt['dropout'],
                    bidirectional=opt['persona_bidirectional'],
                    shared_lt=self.encoder.lt)
        else:
            raise RuntimeError('Unsupported attention target: ' + self.persona_attention)

    def forward(self, xs, ys=None, cands=None, valid_cands=None, ps=None):
        bsz = len(xs)
        if ys is not None:
            # keep track of longest label we've ever seen
            # we'll never produce longer ones than that during prediction
            self.longest_label = max(self.longest_label, ys.size(1))

        enc_out, hidden = self.encoder(xs)

        attn_mask = None
        if self.attn_type != 'none':
            if ps is None or self.persona_attention == 'context':
                # attention over just the context
                attn_mask = xs.ne(0).float()
            else:
                # use personas with attention
                batch = []
                for row in ps:
                    personas = []
                    if len(row) > 0:
                        if self.persona_encoding == 'bow':
                            bow = self.persona_encoder(torch.cat(row).unsqueeze(0))
                            personas.append(self.bow2hsz(bow))
                        elif self.persona_encoding == 'concat':
                            p_encout, _ps_hid = self.persona_encoder(torch.cat(row).unsqueeze(0))
                            personas.append(p_encout.squeeze(0))
                        else:
                            for p in row:
                                p_encout, _ps_hid = self.persona_encoder(p.unsqueeze(0))
                                if self.persona_encoding == 'separate':
                                    personas.append(p_encout.squeeze(0))
                                elif self.persona_encoding in ['max', 'maxsum']:
                                    personas.append(p_encout.max(1)[0])
                                else:
                                    raise RuntimeError()
                    else:
                        personas.append(enc_out.new().resize_(1, self.hszXdirs).fill_(0))
                    # now add persona encoding to the batch
                    if self.persona_encoding == 'maxsum':
                        batch.append(sum(personas)) # 1 x hsz
                    else:
                        batch.append(torch.cat(personas, dim=0))  # num_personas x hsz
                # add padding so same number of personas per batch row
                max_len = max([b.size(0) for b in batch])
                batch = [F.pad(b, (0, 0, 0, max_len - b.size(0)), value=self.NULL_IDX) for b in batch]
                enc_out = torch.cat([b.unsqueeze(0) for b in batch], 0)  # bsz x num_personas x numdir*hsz
                attn_mask = enc_out.select(2, 0).ne(0).float()

        # set up input to decoder
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
