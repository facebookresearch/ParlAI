#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.utils.distributed import is_distributed
from parlai.core.torch_ranker_agent import TorchRankerAgent
from parlai.utils.torch import padded_3d
from parlai.zoo.bert.build import download

from .bert_dictionary import BertDictionaryAgent
from .helpers import (
    get_bert_optimizer,
    BertWrapper,
    BertModel,
    add_common_args,
    surround,
    MODEL_PATH,
)

import os
import torch
from tqdm import tqdm


class BiEncoderRankerAgent(TorchRankerAgent):
    """
    TorchRankerAgent implementation of the biencoder.

    It is a standalone Agent. It might be called by the Both Encoder.
    """

    @staticmethod
    def add_cmdline_args(parser):
        add_common_args(parser)
        parser.set_defaults(encode_candidate_vecs=True)

    def __init__(self, opt, shared=None):
        # download pretrained models
        download(opt['datapath'])
        self.pretrained_path = os.path.join(
            opt['datapath'], 'models', 'bert_models', MODEL_PATH
        )
        opt['pretrained_path'] = self.pretrained_path

        self.clip = -1

        super().__init__(opt, shared)
        # it's easier for now to use DataParallel when
        self.data_parallel = opt.get('data_parallel') and self.use_cuda
        if self.data_parallel and shared is None:
            self.model = torch.nn.DataParallel(self.model)
        if is_distributed():
            raise ValueError('Cannot combine --data-parallel and distributed mode')
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx
        # default one does not average
        self.rank_loss = torch.nn.CrossEntropyLoss(reduce=True, size_average=True)

    def build_model(self):
        return BiEncoderModule(self.opt)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer(
            [self.model],
            self.opt['type_optimization'],
            self.opt['learningrate'],
            fp16=self.opt.get('fp16'),
        )

    def set_vocab_candidates(self, shared):
        """
        Load the tokens from the vocab as candidates.

        self.vocab_candidates will contain a [num_cands] list of strings
        self.vocab_candidate_vecs will contain a [num_cands, 1] LongTensor
        """
        self.opt['encode_candidate_vecs'] = True
        if shared:
            self.vocab_candidates = shared['vocab_candidates']
            self.vocab_candidate_vecs = shared['vocab_candidate_vecs']
            self.vocab_candidate_encs = shared['vocab_candidate_encs']
        else:
            if 'vocab' in (self.opt['candidates'], self.opt['eval_candidates']):
                cands = []
                vecs = []
                for ind in range(1, len(self.dict)):
                    txt = self.dict[ind]
                    cands.append(txt)
                    vecs.append(
                        self._vectorize_text(
                            txt,
                            add_start=True,
                            add_end=True,
                            truncate=self.label_truncate,
                        )
                    )
                self.vocab_candidates = cands
                self.vocab_candidate_vecs = padded_3d([vecs]).squeeze(0)
                print(
                    "[ Loaded fixed candidate set (n = {}) from vocabulary ]"
                    "".format(len(self.vocab_candidates))
                )
                enc_path = self.opt.get('model_file') + '.vocab.encs'
                if os.path.isfile(enc_path):
                    self.vocab_candidate_encs = self.load_candidates(
                        enc_path, cand_type='vocab encodings'
                    )
                else:
                    cand_encs = []
                    vec_batches = [
                        self.vocab_candidate_vecs[i : i + 512]
                        for i in range(0, len(self.vocab_candidate_vecs), 512)
                    ]
                    print(
                        "[ Vectorizing vocab candidates ({} batch(es) of up "
                        "to 512) ]".format(len(vec_batches))
                    )
                    for vec_batch in tqdm(vec_batches):
                        cand_encs.append(self.encode_candidates(vec_batch))
                    self.vocab_candidate_encs = torch.cat(cand_encs, 0)
                    self.save_candidates(
                        self.vocab_candidate_encs, enc_path, cand_type='vocab encodings'
                    )
                if self.use_cuda:
                    self.vocab_candidate_vecs = self.vocab_candidate_vecs.cuda()
                    self.vocab_candidate_encs = self.vocab_candidate_encs.cuda()
            else:
                self.vocab_candidates = None
                self.vocab_candidate_vecs = None
                self.vocab_candidate_encs = None

    def vectorize_fixed_candidates(self, cands_batch):
        """
        Override from TorchRankerAgent.
        """
        return [
            self._vectorize_text(
                cand, add_start=True, add_end=True, truncate=self.label_truncate
            )
            for cand in cands_batch
        ]

    def encode_candidates(self, padded_cands):
        token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
            padded_cands, self.NULL_IDX
        )
        _, embedding_cands = self.model(
            None, None, None, token_idx_cands, segment_idx_cands, mask_cands
        )

        return embedding_cands.cpu().detach()

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

    def score_candidates(self, batch, cand_vecs, cand_encs=None):
        # Encode contexts first
        token_idx_ctxt, segment_idx_ctxt, mask_ctxt = to_bert_input(
            batch.text_vec, self.NULL_IDX
        )
        embedding_ctxt, _ = self.model(
            token_idx_ctxt, segment_idx_ctxt, mask_ctxt, None, None, None
        )

        if cand_encs is not None:
            return embedding_ctxt.mm(cand_encs.t())

        if len(cand_vecs.size()) == 2 and cand_vecs.dtype == torch.long:
            # train time. We compare with all elements of the batch
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cand_vecs, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )
            return embedding_ctxt.mm(embedding_cands.t())

        # predict time with multiple candidates
        if len(cand_vecs.size()) == 3:
            csize = cand_vecs.size()  # batchsize x ncands x sentlength
            cands_idx_reshaped = cand_vecs.view(csize[0] * csize[1], csize[2])
            token_idx_cands, segment_idx_cands, mask_cands = to_bert_input(
                cands_idx_reshaped, self.NULL_IDX
            )
            _, embedding_cands = self.model(
                None, None, None, token_idx_cands, segment_idx_cands, mask_cands
            )
            embedding_cands = embedding_cands.view(
                csize[0], csize[1], -1
            )  # batchsize x ncands x embed_size
            embedding_cands = embedding_cands.transpose(
                1, 2
            )  # batchsize x embed_size x ncands
            embedding_ctxt = embedding_ctxt.unsqueeze(1)  # batchsize x 1 x embed_size
            scores = torch.bmm(
                embedding_ctxt, embedding_cands
            )  # batchsize x 1 x ncands
            scores = scores.squeeze(1)  # batchsize x ncands
            return scores

        # otherwise: cand_vecs should be 2D float vector ncands x embed_size
        return embedding_ctxt.mm(cand_vecs.t())

    def share(self):
        """
        Share model parameters.
        """
        shared = super().share()
        shared['vocab_candidate_encs'] = self.vocab_candidate_encs
        return shared


class BiEncoderModule(torch.nn.Module):
    """
    Groups context_encoder and cand_encoder together.
    """

    def __init__(self, opt):
        super(BiEncoderModule, self).__init__()
        self.context_encoder = BertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['out_dim'],
            add_transformer_layer=opt['add_transformer_layer'],
            layer_pulled=opt['pull_from_layer'],
            aggregation=opt['bert_aggregation'],
        )
        self.cand_encoder = BertWrapper(
            BertModel.from_pretrained(opt['pretrained_path']),
            opt['out_dim'],
            add_transformer_layer=opt['add_transformer_layer'],
            layer_pulled=opt['pull_from_layer'],
            aggregation=opt['bert_aggregation'],
        )

    def forward(
        self,
        token_idx_ctxt,
        segment_idx_ctxt,
        mask_ctxt,
        token_idx_cands,
        segment_idx_cands,
        mask_cands,
    ):
        embedding_ctxt = None
        if token_idx_ctxt is not None:
            embedding_ctxt = self.context_encoder(
                token_idx_ctxt, segment_idx_ctxt, mask_ctxt
            )
        embedding_cands = None
        if token_idx_cands is not None:
            embedding_cands = self.cand_encoder(
                token_idx_cands, segment_idx_cands, mask_cands
            )
        return embedding_ctxt, embedding_cands


def to_bert_input(token_idx, null_idx):
    """
    token_idx is a 2D tensor int.

    return token_idx, segment_idx and mask
    """
    segment_idx = token_idx * 0
    mask = token_idx != null_idx
    # nullify elements in case self.NULL_IDX was not 0
    token_idx = token_idx * mask.long()
    return token_idx, segment_idx, mask
