#!/usr/bin/env python3

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F
import torch

from .gru import BayesianGRU
from .loadstates import (
    load_dictionary,
    load_emb_params,
    make_emb_state_dict,
    load_rnn_params,
    make_bayesian_state_dict,
    make_gru_state_dict,
)


class Mlb(nn.Module):
    def __init__(self, opt, vocab):
        super(Mlb, self).__init__()
        self.opt = opt
        self.training = self.opt.get('datatype').startswith('train')
        self.dict = vocab
        self.vocab_words = self.dict.tok2ind.keys()
        self.vocab_answers = self.dict.ans2ind.keys()
        self.num_classes = len(self.vocab_answers)
        # Modules
        self.embedding = nn.Embedding(
            num_embeddings=len(self.dict.tok2ind),
            embedding_dim=620,
            padding_idx=self.dict.tok2ind[self.dict.null_token],
            sparse=False
        )

        if self.opt['use_bayesian']:
            self.rnn = BayesianGRU(620,
                                   self.opt['dim_q'],
                                   dropout=self.opt['dropout_st'])
        else:
            self.rnn = nn.GRU(input_size=620,
                              hidden_size=self.opt['dim_q'],
                              batch_first=True,
                              dropout=self.opt['dropout_st'])

    def process_lengths(self, input):
        max_length = input.size(1)
        if input.size(0) != 1:
            sub = input.eq(0).sum(1).squeeze(0)
        else:
            sub = input.eq(0).sum(1)
        lengths = list(max_length - sub)
        return lengths

    def select_last(self, x, lengths):
        batch_size = x.size(0)
        mask = x.new().resize_as_(x).fill_(0)
        for i in range(batch_size):
            mask[i][lengths[i] - 1].fill_(1)
        x = x.mul(mask)
        x = x.sum(1).view(batch_size, self.opt['dim_q'])
        return x

    def _classif(self, x):
        x = getattr(F, self.opt['activation_cls'])(x)
        x = F.dropout(x, p=self.opt['dropout_cls'], training=self.training)
        x = self.linear_classif(x)
        return x

    def forward_st(self, input, lengths=None):
        if lengths is None:
            lengths = self.process_lengths(input)
        x = self.embedding(input)
        max_length = max(lengths)
        x, hn = self.rnn(x, max_length=max_length)  # seq2seq
        if lengths:
            x = self.select_last(x, lengths)
        return x

    def save(self, path=None):
        path = self.opt.get('model_file', None) if path is None else path

        if path and hasattr(self, 'embedding'):
            print("[ saving model: " + path + " ]")

            model = {
                'model': self.state_dict(),
                'optim': self.optim.state_dict(),
                'opt': self.opt
            }
            with open(path, 'wb') as write:
                torch.save(model, write)

    def set_states(self, states):
        """Set the state dicts of the modules from saved states."""
        self.load_state_dict(states['model'])

    def set_init_states(self):
        """Set the initial state dicts of the modules from saved states."""
        dictionary = load_dictionary(self.opt['download_path'])
        parameters = load_emb_params(self.opt['download_path'])
        state_dict = make_emb_state_dict(dictionary,
                                         parameters,
                                         self.dict.ind2tok.values())
        self.embedding.load_state_dict(state_dict)
        parameters = load_rnn_params(self.opt['download_path'])
        if self.opt['use_bayesian']:
            state_dict = make_bayesian_state_dict(parameters)
        else:
            state_dict = make_gru_state_dict(parameters)
        self.rnn.load_state_dict(state_dict)
        return self.rnn

    def get_optim(self):
        optim_class = torch.optim.Adam
        self.optim = optim_class(filter(lambda p: p.requires_grad,
                                        self.parameters()),
                                 lr=self.opt['lr'])
        if self.states:
            self.optim.load_state_dict(self.states['optim'])

        return self.optim


class MlbNoAtt(Mlb):
    def __init__(self, opt, vocab, states):
        super(MlbNoAtt, self).__init__(opt, vocab)
        self.linear_v = nn.Linear(self.opt['dim_v'], self.opt['dim_h'])
        self.linear_q = nn.Linear(self.opt['dim_q'], self.opt['dim_h'])
        self.linear_classif = nn.Linear(self.opt['dim_h'], self.num_classes)
        self.states = states

        if self.states:
            # set loaded states if applicable
            self.set_states(self.states)
        else:
            self.set_init_states()

    def forward(self, input_v, input_q):
        x_q = self.forward_st(input_q)
        x = self.forward_fusion(input_v, x_q)
        x = self._classif(x)
        return x

    def forward_fusion(self, input_v, input_q):
        # visual (cnn features)
        x_v = F.dropout(input_v,
                        p=self.opt['dropout_v'],
                        training=self.training)
        x_v = self.linear_v(x_v)
        x_v = getattr(F, self.opt['activation_v'])(x_v)
        # question (rnn features)
        x_q = F.dropout(input_q,
                        p=self.opt['dropout_q'],
                        training=self.training)
        x_q = self.linear_q(x_q)
        x_q = getattr(F, self.opt['activation_q'])(x_q)
        # hadamard product
        x_mm = torch.mul(x_q, x_v)
        return x_mm


class MlbAtt(Mlb):
    def __init__(self, opt, vocab, states):
        super(MlbAtt, self).__init__(opt, vocab)
        self.conv_v_att = nn.Conv2d(self.opt['dim_v'],
                                    self.opt['dim_att_h'],
                                    1, 1)
        self.linear_q_att = nn.Linear(self.opt['dim_q'], self.opt['dim_att_h'])
        self.conv_att = nn.Conv2d(self.opt['dim_att_h'],
                                  self.opt['num_glimpses'],
                                  1, 1)
        if self.opt['original_att']:
            self.linear_v_fusion = nn.Linear(self.opt['dim_v'] *
                                             self.opt['num_glimpses'],
                                             self.opt['dim_h'])
            self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                             self.opt['dim_h'])
            self.linear_classif = nn.Linear(self.opt['dim_h'],
                                            self.num_classes)
        else:
            self.list_linear_v_fusion = nn.ModuleList(
                [nn.Linear(self.opt['dim_v'], self.opt['dim_h'])
                 for i in range(self.opt['num_glimpses'])]
            )
            self.linear_q_fusion = nn.Linear(self.opt['dim_q'],
                                             self.opt['dim_h'] *
                                             self.opt['num_glimpses'])
            self.linear_classif = nn.Linear(
                self.opt['dim_h'] * self.opt['num_glimpses'],
                self.num_classes
            )

        self.states = states
        if self.states:
            # set loaded states if applicable
            self.set_states(self.states)
        else:
            self.set_init_states()

    def forward(self, input_v, input_q):
        x_q = self.forward_st(input_q)
        list_v_att = self.forward_attention(input_v, x_q)
        x = self.forward_glimpses(list_v_att, x_q)
        x = self._classif(x)
        return x

    def forward_attention(self, input_v, x_q_vec):
        batch_size = input_v.size(0)
        width = input_v.size(2)
        height = input_v.size(3)

        # Process visual before fusion
        x_v = input_v
        x_v = F.dropout(x_v,
                        p=self.opt['dropout_att_v'],
                        training=self.training)
        x_v = self.conv_v_att(x_v)
        x_v = getattr(F, self.opt['activation_att_v'])(x_v)
        x_v = x_v.view(batch_size, self.opt['dim_att_h'], width * height)
        x_v = x_v.transpose(1, 2)

        # Process question before fusion
        x_q = F.dropout(x_q_vec,
                        p=self.opt['dropout_att_q'],
                        training=self.training)
        x_q = self.linear_q_att(x_q)
        x_q = getattr(F, self.opt['activation_att_q'])(x_q)
        x_q = x_q.view(batch_size, 1, self.opt['dim_att_h'])
        x_q = x_q.expand(batch_size, width * height, self.opt['dim_att_h'])

        # First multimodal fusion
        x_att = self.forward_fusion_att(x_v, x_q)
        x_att = getattr(F, self.opt['activation_att_mm'])(x_att)

        # Process attention vectors
        x_att = F.dropout(x_att,
                          p=self.opt['dropout_att_mm'],
                          training=self.training)
        x_att = x_att.view(batch_size, width, height, self.opt['dim_att_h'])
        x_att = x_att.transpose(2, 3).transpose(1, 2)
        x_att = self.conv_att(x_att)
        x_att = x_att.view(batch_size,
                           self.opt['num_glimpses'],
                           width * height)
        list_att_split = torch.split(x_att, 1, dim=1)
        list_att = []
        for x_att in list_att_split:
            x_att = x_att.contiguous()
            x_att = x_att.view(batch_size, width * height)
            x_att = F.softmax(x_att)
            list_att.append(x_att)

        # Apply attention vectors to input_v
        x_v = input_v.view(batch_size, self.opt['dim_v'], width * height)
        x_v = x_v.transpose(1, 2)

        list_v_att = []
        for x_att in list_att:
            x_att = x_att.view(batch_size, width * height, 1)
            x_att = x_att.expand(batch_size, width * height, self.opt['dim_v'])
            x_v_att = torch.mul(x_att, x_v)
            x_v_att = x_v_att.sum(1)
            x_v_att = x_v_att.view(batch_size, self.opt['dim_v'])
            list_v_att.append(x_v_att)

        return list_v_att

    def forward_glimpses(self, list_v_att, x_q_vec):
        # Process visual for each glimpses
        list_v = []
        if self.opt['original_att']:
            x_v = torch.cat(list_v_att, 1)
            x_v = F.dropout(x_v,
                            p=self.opt['dropout_v'],
                            training=self.training)
            x_v = self.linear_v_fusion(x_v)
            x_v = getattr(F, self.opt['activation_v'])(x_v)
        else:
            for glimpse_id, x_v_att in enumerate(list_v_att):
                x_v = F.dropout(x_v_att,
                                p=self.opt['dropout_v'],
                                training=self.training)
                x_v = self.list_linear_v_fusion[glimpse_id](x_v)
                x_v = getattr(F, self.opt['activation_v'])(x_v)
                list_v.append(x_v)
            x_v = torch.cat(list_v, 1)

        # Process question
        x_q = F.dropout(x_q_vec,
                        p=self.opt['dropout_q'],
                        training=self.training)
        x_q = self.linear_q_fusion(x_q)
        x_q = getattr(F, self.opt['activation_q'])(x_q)

        # Second multimodal fusion
        x = self.forward_fusion_cls(x_v, x_q)
        return x

    def forward_fusion_att(self, input_v, input_q):
        x_att = torch.mul(input_v, input_q)
        return x_att

    def forward_fusion_cls(self, input_v, input_q):
        x_att = torch.mul(input_v, input_q)
        return x_att
