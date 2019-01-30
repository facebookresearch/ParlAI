#!/usr/bin/env python3

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. An additional grant
# of patent rights can be found in the PATENTS file in the same directory.
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .bert_dictionary import BertDictionaryAgent
from .helpers import (get_bert_optimizer, BertWrapper, BertModel,
                      add_common_args, surround)
import torch
import json


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
        if all(f in self.opt for f in ["num_samples", "num_epochs", "batchsize"]):
            total_iterations = self.opt["num_samples"] * \
                self.opt["num_epochs"] / self.opt["batchsize"]
            self.optimizer = get_bert_optimizer([self.model],
                                                self.opt["type_optimization"],
                                                total_iterations,
                                                0.05,  # 5% scheduled warmup.
                                                self.opt["learningrate"])

    def score_candidates(self, batch, cand_vecs):
        # concatenate text and candidates (not so easy)
        # unpad and break
        nb_cands = cand_vecs.size()[1]
        size_batch = cand_vecs.size()[0]
        text_vec = pad_left(batch.text_vec, self.NULL_IDX)
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

    def vectorize(self, obs, add_start=True, add_end=True, truncate=None,
                  split_lines=False):
        return super().vectorize(
            obs,
            add_start=True,
            add_end=True,
            truncate=self.truncate)

    def _set_text_vec(self, obs, truncate, split_lines):
        super()._set_text_vec(obs, truncate, split_lines)
        # concatenate the [CLS] and [SEP] tokens
        if obs is not None and "text_vec" in obs:
            obs["text_vec"] = surround(obs["text_vec"], self.START_IDX, self.END_IDX)
        return obs

    def receive_metrics(self, metrics_dict):
        """ Inibiting the scheduler.
        """
        pass


def pad_left(context_idx, null_idx):
    """ Take a 2D padded to the right and pad it to the left instead.
    """
    new_tensor = context_idx * 0 + null_idx
    num_pads = torch.sum(context_idx == null_idx, 1)
    for i, vec in enumerate(context_idx):
        offset = int(num_pads[i].cpu().item())
        if offset == 0:
            new_tensor[i] = vec
        else:
            new_tensor[i, offset:] = vec[0:-offset]
    return new_tensor
