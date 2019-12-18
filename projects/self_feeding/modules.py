#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from parlai.agents.transformer.modules import TransformerEncoder


class SelfFeedingModel(nn.Module):
    @classmethod
    def add_cmdline_args(cls, argparser):
        model = argparser.add_argument_group('Self Feeding Model')

        model.add_argument(
            '-shl',
            '--sat-head-layers',
            type=int,
            default=1,
            help="The number of linear layers in the " "satisfaction task head",
        )
        model.add_argument(
            '-sfeeemb',
            '--share-fee-embeddings',
            type='bool',
            default=True,
            help="If True, the feedback task shares " "the dialog embeddings",
        )
        model.add_argument(
            '-sfeexenc',
            '--share-fee-x-encoder',
            type='bool',
            default=True,
            help="If True, the feedback task shares " "the dialog x encoder",
        )
        model.add_argument(
            '-sfeeyenc',
            '--share-fee-y-encoder',
            type='bool',
            default=True,
            help="If True, the feedback task shares " "the dialog y encoder",
        )
        model.add_argument(
            '-ssatemb',
            '--share-sat-embeddings',
            type='bool',
            default=False,
            help="If True, the satisfaction task shares " "the dialog embeddings",
        )
        model.add_argument(
            '-ssatenc',
            '--share-sat-encoder',
            type='bool',
            default=False,
            help="If True, the satisfaction task shares the dialog " "encoder",
        )

    def __init__(self, opt, dictionary):
        super().__init__()
        self.opt = opt
        self.pad_idx = dictionary[dictionary.null_token]
        self.vocab_size = len(dictionary)

        # Build dialog
        self.dia_embeddings = self.init_embeddings()
        self.x_dia_encoder = self.build_encoder(opt, self.dia_embeddings)
        self.x_dia_head = nn.Dropout(p=0)

        self.y_dia_encoder = self.build_encoder(opt, self.dia_embeddings)
        self.y_dia_head = nn.Dropout(p=0)

        # Only build the parts of the network you will be using
        # This saves space (nbd) and prevents conflicts when loading
        # Build feedback
        if 'feedback' in self.opt['subtasks']:
            if self.opt['share_fee_embeddings']:
                self.fee_embeddings = self.dia_embeddings
            else:
                self.fee_embeddings = self.init_embeddings()
            if self.opt['share_fee_x_encoder']:
                self.x_fee_encoder = self.x_dia_encoder
            else:
                self.x_fee_encoder = self.build_encoder(opt, self.fee_embeddings)
            self.x_fee_head = nn.Dropout(p=0)

            if self.opt['share_fee_y_encoder']:
                self.y_fee_encoder = self.y_dia_encoder
            else:
                self.y_fee_encoder = self.build_encoder(opt, self.fee_embeddings)
            self.y_fee_head = nn.Dropout(p=0)

        # Build satisfaction
        if 'satisfaction' in self.opt['subtasks']:
            if self.opt['share_sat_embeddings']:
                self.sat_embeddings = self.dia_embeddings
            else:
                self.sat_embeddings = self.init_embeddings()
            if self.opt['share_sat_encoder']:
                self.x_sat_encoder = self.x_dia_encoder
            else:
                self.x_sat_encoder = self.build_encoder(opt, self.sat_embeddings)
            self.x_sat_head = self.build_head(
                opt, outdim=1, num_layers=self.opt['sat_head_layers']
            )

    def forward(self):
        raise NotImplementedError

    def score_dialog(self, x_vecs, y_vecs):
        x_enc = self.x_dia_head(self.x_dia_encoder(x_vecs))

        if y_vecs.dtype == torch.float32:
            # Assume candidates have already been encoded (e.g., in interactive mode)
            y_enc = y_vecs
        elif y_vecs.dtype == torch.int64:
            # Assume candidates have only been vectorized
            y_enc = self.encode_dia_y(y_vecs)
        else:
            raise Exception("Unsupported type for cands: {}".format(type(y_vecs)))

        return self.score_similarity(x_enc, y_enc)

    def encode_dia_y(self, y_vecs):
        """
        Encodes a tensor of vectorized candidates.

        :param y_vecs: a [bs, seq_len] or [bs, num_cands, seq_len](?) of vectorized
            candidates
        """
        if y_vecs.dim() == 2:
            y_enc = self.y_dia_head(self.y_dia_encoder(y_vecs))
        elif y_vecs.dim() == 3:
            oldshape = y_vecs.shape
            y_vecs = y_vecs.reshape(oldshape[0] * oldshape[1], oldshape[2])
            y_enc = self.y_dia_head(self.y_dia_encoder(y_vecs))
            y_enc = y_enc.reshape(oldshape[0], oldshape[1], -1)
        return y_enc

    def score_feedback(self, x_vecs, y_vecs):
        x_enc = self.x_fee_head(self.x_fee_encoder(x_vecs))
        y_enc = self.y_fee_head(self.y_fee_encoder(y_vecs))
        return self.score_similarity(x_enc, y_enc)

    def score_satisfaction(self, x_vecs):
        return torch.sigmoid(self.x_sat_head(self.x_sat_encoder(x_vecs))).squeeze(1)

    def score_similarity(self, context_h, cand_h):
        """
        Returns the dot product of encoded contexts and encoded candidates.
        """
        if self.opt['normalize_sent_emb']:
            context_h /= context_h.norm(2, dim=1, keepdim=True)
            cand_h /= cand_h.norm(2, dim=1, keepdim=True)

        if cand_h.dim() == 2:
            scores = torch.matmul(context_h, cand_h.t())
        elif cand_h.dim() == 3:
            scores = torch.bmm(context_h.unsqueeze(1), cand_h.transpose(1, 2)).squeeze(
                1
            )
        else:
            raise RuntimeError(
                'Unexpected candidate dimensions {}' ''.format(cand_h.dim())
            )

        return scores

    def init_embeddings(self):
        embeddings = nn.Embedding(
            self.vocab_size, self.opt['embedding_size'], padding_idx=self.pad_idx
        )
        nn.init.normal_(
            embeddings.weight, mean=0, std=self.opt['embedding_size'] ** -0.5
        )
        nn.init.constant_(embeddings.weight[self.pad_idx], 0)
        return embeddings

    def build_encoder(self, opt, embeddings):
        return TransformerEncoder(
            n_heads=opt['n_heads'],
            n_layers=opt['n_layers'],
            embedding_size=opt['embedding_size'],
            ffn_size=opt['ffn_size'],
            vocabulary_size=self.vocab_size,
            embedding=embeddings,
            attention_dropout=opt['attention_dropout'],
            relu_dropout=opt['relu_dropout'],
            padding_idx=self.pad_idx,
            learn_positional_embeddings=opt.get('learn_positional_embeddings', False),
            embeddings_scale=opt['embeddings_scale'],
        )

    def build_head(self, opt, outdim=1, num_layers=1):
        dim = self.opt['embedding_size']
        modules = []
        for _ in range(num_layers - 1):
            modules.append(nn.Linear(dim, dim))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(dim, outdim))
        return nn.Sequential(*modules)


class Identity(nn.Module):
    def forward(self, x):
        return x
