#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.distributed_utils import is_distributed
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.zoo.bert.build import download

from .bert_dictionary import BertDictionaryAgent
from .helpers import (
    BertWrapper,
    BertModel,
    get_bert_optimizer,
    add_common_args,
    surround,
    MODEL_PATH,
)

import os
import torch


def concat_without_padding(text_idx, cand_idx, use_cuda, null_idx=0):
    """ if text_idx = [[1, 2, 3, 4, 0, 0  ]]
            and cand_idx = [[5, 6, 7, 8, 0, 0 ]]
            then result = (tokens, segments) where
            tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0]]
            segments = [[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0]]
        """
    assert text_idx.size(0) == cand_idx.size(0)
    assert len(text_idx.size()) == 2
    assert len(cand_idx.size()) == 2
    segments_idx = [0, 1]
    text_idx = text_idx.cpu()
    cand_idx = cand_idx.cpu()
    cand_len = cand_idx.size(1)
    concat_len = text_idx.size(1) + cand_idx.size(1)
    tokens = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    segments = text_idx.new_zeros(text_idx.size(0), concat_len) + null_idx
    for i in range(len(tokens)):
        non_nuls = torch.sum(text_idx[i, :] != null_idx)
        tokens[i, 0:non_nuls] = text_idx[i, 0:non_nuls]
        segments[i, 0:non_nuls] = segments_idx[0]
        tokens[i, non_nuls : non_nuls + cand_len] = cand_idx[i, :]
        segments[i, non_nuls : non_nuls + cand_len] = segments_idx[1]
    if use_cuda:
        tokens = tokens.cuda()
        segments = segments.cuda()
    return tokens, segments


class CrossEncoderRankerAgent(TorchRankerAgent):
    """ TorchRankerAgent implementation of the crossencoder.
        It is a standalone Agent. It might be called by the Both Encoder.
    """

    @staticmethod
    def add_cmdline_args(parser):
        add_common_args(parser)
        parser.set_defaults(encode_candidate_vecs=False, candidates='inline')

    def __init__(self, opt, shared=None):
        # download pretrained models
        download(opt['datapath'])
        self.pretrained_path = os.path.join(
            opt['datapath'], 'models', 'bert_models', MODEL_PATH
        )

        super().__init__(opt, shared)
        # it's easier for now to use DataParallel when
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel and shared is None:
            self.model = torch.nn.DataParallel(self.model)
        if is_distributed():
            raise ValueError('Cannot combine --data-parallel and distributed mode')
        self.clip = -1
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx

    def build_model(self):
        return BertWrapper(
            BertModel.from_pretrained(self.pretrained_path),
            1,
            add_transformer_layer=self.opt['add_transformer_layer'],
            layer_pulled=self.opt['pull_from_layer'],
            aggregation=self.opt['bert_aggregation'],
        )

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer(
            [self.model],
            self.opt['type_optimization'],
            self.opt['learningrate'],
            fp16=self.opt.get('fp16'),
        )

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        # concatenate text and candidates (not so easy)
        # unpad and break
        nb_cands = cand_vecs.size()[1]
        size_batch = cand_vecs.size()[0]
        text_vec = batch.text_vec

        tokens_context = (
            text_vec.unsqueeze(1)
            .expand(-1, nb_cands, -1)
            .contiguous()
            .view(nb_cands * size_batch, -1)
        )
        segments_context = tokens_context * 0

        # remove the start token ["CLS"] from candidates
        tokens_cands = cand_vecs.view(nb_cands * size_batch, -1)
        segments_cands = tokens_cands * 0 + 1
        all_tokens, all_segments = concat_without_padding(
            tokens_context, tokens_cands, self.use_cuda, self.NULL_IDX
        )
        all_mask = all_tokens != self.NULL_IDX
        all_tokens *= all_mask.long()
        scores = self.model(all_tokens, all_segments, all_mask)
        return scores.view(size_batch, nb_cands)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def _set_text_vec(self, *args, **kwargs):
        obs = super()._set_text_vec(*args, **kwargs)
        # concatenate the [CLS] and [SEP] tokens
        if (
            obs is not None
            and 'text_vec' in obs
            and 'added_start_end_tokens' not in obs
        ):
            obs.force_set(
                'text_vec', surround(obs['text_vec'], self.START_IDX, self.END_IDX)
            )
            obs['added_start_end_tokens'] = True
        return obs
