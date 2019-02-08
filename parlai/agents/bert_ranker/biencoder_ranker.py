#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.core.torch_ranker_agent import TorchRankerAgent
from .bert_dictionary import BertDictionaryAgent
from .helpers import (get_bert_optimizer, BertWrapper, BertModel,
                      add_common_args, surround)
from parlai.core.utils import padded_3d
from parlai.core.distributed_utils import is_distributed
import torch
import tqdm


class BiEncoderRankerAgent(TorchRankerAgent):
    """ TorchRankerAgent implementation of the biencoder.
        It is a standalone Agent. It might be called by the Both Encoder.
    """

    @staticmethod
    def add_cmdline_args(parser):
        add_common_args(parser)

    def __init__(self, opt, shared=None):
        opt['rank_candidates'] = True
        opt['candidates'] = "batch"
        if opt.get('eval_candidates', None) is None:
            opt['eval_candidates'] = "inline"
        self.clip = -1
        super().__init__(opt, shared)
        # it's easier for now to use DataParallel when
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel:
            self.model = torch.nn.DataParallel(self.model)
        if is_distributed():
            raise ValueError('Cannot combine --data-parallel and distributed mode')
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx
        # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)

    def build_model(self):
        self.model = BiEncoderModule(self.opt)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer([self.model],
                                            self.opt["type_optimization"],
                                            self.opt["learningrate"])

    def make_candidate_vecs(self, cands):
        cand_batches = [cands[i:i + 200] for i in range(0, len(cands), 200)]
        cand_vecs = []
        for batch in tqdm.tqdm(cand_batches,
                               desc="[ Vectorizing fixed candidates set from "
                                    "({} batch(es) of up to 200) ]"
                                    "".format(len(cand_batches))):
            token_idx = [self._vectorize_text(cand, add_start=True, add_end=True,
                                              truncate=self.opt["label_truncate"])
                         for cand in batch]
            padded_input = padded_3d([token_idx]).squeeze(0)
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                padded_input, self.NULL_IDX)
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands)
            cand_vecs.append(embedding_cands.cpu().detach())
        return torch.cat(cand_vecs, 0)

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

    def score_candidates(self, batch, cand_vecs):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            batch.text_vec, self.NULL_IDX)
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
            None, None, None)
        if len(cand_vecs.size()) == 2 and cand_vecs.dtype == torch.long:
            # train time. We compare with all elements of the batch
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX)
            _, embedding_cands = self.model(
                None, None, None,
                token_idx_cands, segment_idx_cands, mask_cands)
            return embedding_ctxt.mm(embedding_cands.t())

        # predict time with multiple candidates
        if len(cand_vecs.size()) == 3:
            csize = cand_vecs.size()  # batchsize x ncands x sentlength
            cands_idx_reshaped = cand_vecs.view(csize[0] * csize[1], csize[2])
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cands_idx_reshaped, self.NULL_IDX)
            _, embedding_cands = self.model(
                None, None, None,
                token_idx_cands, segment_idx_cands, mask_cands)
            embedding_cands = embedding_cands.view(
                csize[0], csize[1], -1)  # batchsize x ncands x embed_size
            embedding_cands = embedding_cands.transpose(
                1, 2)  # batchsize x embed_size x ncands
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            scores = torch.bmm(embedding_ctxt,
                               embedding_cands)  # batchsize x 1 x ncands
            scores = scores.squeeze(1)  # batchsize x ncands
            return scores

        # otherwise: cand_vecs should be 2D float vector ncands x embed_size
        return embedding_ctxt.mm(cand_vecs.t())


class BiEncoderModule(torch.nn.Module):
    """ Groups context_encoder and cand_encoder together.
    """

    def __init__(self, opt):
        super(BiEncoderModule, self).__init__()
        self.context_encoder = BertWrapper(
            BertModel.from_pretrained(
                opt["pretrained_bert_path"]),
            opt["out_dim"],
            add_transformer_layer=opt["add_transformer_layer"],
            layer_pulled=opt["pull_from_layer"])
        self.cand_encoder = BertWrapper(
            BertModel.from_pretrained(
                opt["pretrained_bert_path"]),
            opt["out_dim"],
            add_transformer_layer=opt["add_transformer_layer"],
            layer_pulled=opt["pull_from_layer"])

    def forward(self, token_idx_ctxt, segment_idx_ctxt, mask_ctxt,
                token_idx_cands, segment_idx_cands, mask_cands):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt)
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands)
        return embedding_ctxt, embedding_cands


def to_bert_input(token_idx, null_idx):
    """ token_idx is a 2D tensor int.
        return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = (token_idx != null_idx).long()
    token_idx = token_idx * mask  # nullify elements in case self.NULL_IDX was not 0
    return token_idx, segment_idx, mask
