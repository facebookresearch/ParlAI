#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from .bert_dictionary import BertDictionaryAgent
from .bi_encoder_ranker import BiEncoderRankerAgent
from .cross_encoder_ranker import CrossEncoderRankerAgent
from .helpers import add_common_args

from parlai.core.torch_agent import TorchAgent, Output, Batch


class BothEncoderRankerAgent(TorchAgent):
    """
    A Bi Encoder followed by a Cross Encoder.

    Although it's trainable by itself, I'd recommend training the crossencoder and the
    biencoder separately which can be done in parallel or sequentially and thus
    requiring less memory on the GPU.
    """

    @staticmethod
    def add_cmdline_args(parser):
        add_common_args(parser)
        parser = parser.add_argument_group('Bert Ranker Arguments')
        parser.add_argument(
            '--biencoder-model-file',
            type=str,
            default=None,
            help='path to biencoder model. Default to model-file_bi',
        )
        parser.add_argument(
            '--biencoder-top-n',
            type=int,
            default=10,
            help='default number of elements to keep from the biencoder response',
        )
        parser.add_argument(
            '--crossencoder-model-file',
            type=str,
            default=None,
            help='path to crossencoder model. Default to model-file_cross',
        )
        parser.add_argument(
            '--crossencoder-batchsize',
            type=int,
            default=-1,
            help='crossencoder will be fed those many elements at train or eval time.',
        )
        parser.set_defaults(
            encode_candidate_vecs=True, dict_maxexs=0  # skip building dictionary
        )

    def __init__(self, opt, shared=None):
        opt['lr_scheduler'] = 'none'
        self.path_biencoder = opt.get('biencoder_model_file', None)
        if self.path_biencoder is None:
            self.path_biencoder = opt['model_file'] + '_bi'
        self.path_crossencoder = opt.get('crossencoder_model_file', None)
        if self.path_crossencoder is None:
            self.path_crossencoder = opt['model_file'] + '_cross'
        self.top_n_bi = opt['biencoder_top_n']

        super().__init__(opt, shared)
        self.NULL_IDX = self.dict.pad_idx
        self.START_IDX = self.dict.start_idx
        self.END_IDX = self.dict.end_idx
        if shared is None:
            opt_biencoder = dict(opt)
            opt_biencoder['model_file'] = self.path_biencoder
            self.biencoder = BiEncoderRankerAgent(opt_biencoder)
            opt_crossencoder = dict(opt)
            opt_crossencoder['model_file'] = self.path_crossencoder
            opt_crossencoder['batchsize'] = opt['batchsize']
            opt_crossencoder['eval_candidates'] = 'inline'
            if opt['crossencoder_batchsize'] != -1:
                opt_crossencoder['batchsize'] = opt['crossencoder_batchsize']
            self.crossencoder_batchsize = opt_crossencoder['batchsize']
            self.crossencoder = CrossEncoderRankerAgent(opt_crossencoder)

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def train_step(self, batch):
        self.biencoder.train_step(batch)
        # the crossencoder requires batches that are smaller.
        # split the batch into smaller pieces
        outc = []
        step = self.crossencoder_batchsize
        for start in range(0, len(batch.text_vec), step):
            mbatch = Batch(
                text_vec=batch.text_vec[start : start + step],
                label_vec=batch.label_vec[start : start + step],
                candidate_vecs=batch.candidate_vecs[start : start + step],
                candidates=batch.candidates[start : start + step],
            )
            outc.append(self.crossencoder.train_step(mbatch))
        return Output(text=[text for out in outc for text in out.text])

    def eval_step(self, batch):
        """
        We pass the batch first in the biencoder, then filter with crossencoder.
        """
        output_biencoder = self.biencoder.eval_step(batch)
        if output_biencoder is None:
            return None
        new_candidate_vecs = [
            self.biencoder.vectorize_fixed_candidates(cands[0 : self.top_n_bi])
            for cands in output_biencoder.text_candidates
        ]
        new_candidates = [
            [c for c in cands[0 : self.top_n_bi]]
            for cands in output_biencoder.text_candidates
            if cands is not None
        ]
        copy_batch = Batch(
            text_vec=batch.text_vec,
            candidate_vecs=new_candidate_vecs,
            candidates=new_candidates,
        )
        return self.crossencoder.eval_step(copy_batch)

    def save(self, path=None):
        self.biencoder.save(self.path_biencoder)
        self.crossencoder.save(self.path_crossencoder)

    def load(self, path):
        self.biencoder.load(self.path_biencoder)
        self.crossencoder.load(self.path_crossencoder)
