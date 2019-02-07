#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .bert_dictionary import BertDictionaryAgent
from .helpers import (BertWrapper, BertModel, get_bert_optimizer,
                      add_common_args, surround)
from parlai.core.distributed_utils import is_distributed
import torch


class CrossEncoderRankerAgent(TorchRankerAgent):
    """ TorchRankerAgent implementation of the crossencoder.
        It is a standalone Agent. It might be called by the Both Encoder.
    """

    @staticmethod
    def add_cmdline_args(parser):
        add_common_args(parser)

    def __init__(self, opt, shared=None):
        opt['rank_candidates'] = True
        super().__init__(opt, shared)
        # it's easier for now to use DataParallel when
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        if is_distributed():
            raise ValueError('Cannot combine --data-parallel and distributed mode')
        self.clip = -1
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx

    def build_model(self):
        self.model = BertWrapper(
            BertModel.from_pretrained(
                self.opt["pretrained_bert_path"]),
            1,
            add_transformer_layer=self.opt["add_transformer_layer"],
            layer_pulled=self.opt["pull_from_layer"])

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer([self.model],
                                            self.opt["type_optimization"],
                                            self.opt["learningrate"])

    def score_candidates(self, batch, cand_vecs):
        # concatenate text and candidates (not so easy)
        # unpad and break
        nb_cands = cand_vecs.size()[1]
        size_batch = cand_vecs.size()[0]
        text_vec = batch.text_vec
        tokens_context = text_vec.unsqueeze(
            1).expand(-1, nb_cands, -1).contiguous().view(nb_cands * size_batch, -1)
        segments_context = tokens_context * 0

        # remove the start token ["CLS"] from candidates
        tokens_cands = cand_vecs.view(nb_cands * size_batch, -1)
        segments_cands = tokens_cands * 0 + 1
        all_tokens = torch.cat([tokens_context, tokens_cands], 1)
        all_segments = torch.cat([segments_context, segments_cands], 1)
        all_mask = (all_tokens != self.NULL_IDX).long()
        all_tokens *= all_mask
        scores = self.model(all_tokens, all_segments, all_mask)
        return scores.view(size_batch, nb_cands)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def vectorize(self, obs, add_start=True, add_end=True, split_lines=False,
                  text_truncate=None, label_truncate=None):
        return super().vectorize(
            obs,
            add_start=True,
            add_end=True,
            text_truncate=self.text_truncate,
            label_truncate=self.label_truncate)

    def _set_text_vec(self, obs, truncate, split_lines):
        super()._set_text_vec(obs, truncate, split_lines)
        # concatenate the [CLS] and [SEP] tokens
        if obs is not None and "text_vec" in obs:
            obs["text_vec"] = surround(obs["text_vec"], self.START_IDX, self.END_IDX)
        return obs
