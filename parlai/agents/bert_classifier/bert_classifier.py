#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from parlai.agents.bert_ranker.bert_dictionary import BertDictionaryAgent
from parlai.agents.bert_ranker.helpers import (
    BertWrapper,
    get_bert_optimizer,
    MODEL_PATH
)
from parlai.core.torch_agent import History
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.zoo.bert.build import download

from collections import deque
import os
import torch
try:
    from pytorch_pretrained_bert import BertModel
except ImportError:
    raise Exception(("BERT rankers needs pytorch-pretrained-BERT installed. \n "
                     "pip install pytorch-pretrained-bert"))


class BertClassifierHistory(History):
    def __init__(self, opt, **kwargs):
        self.sep_last_utt = opt.get('sep_last_utt', False)
        super().__init__(opt, **kwargs)

    def get_history_vec(self):
        """Override from parent class to possibly add [SEP] token."""
        if not self.sep_last_utt or len(self.history_vecs) <= 1:
            return super().get_history_vec()

        history = deque(maxlen=self.max_len)
        for vec in self.history_vecs[:-1]:
            history.extend(vec)
            history.extend(self.delimiter_tok)
        history.extend([self.dict.end_idx])  # add [SEP] token
        history.extend(self.history_vecs[-1])

        return history


class BertClassifierAgent(TorchClassifierAgent):
    """
    Classifier based on Hugging Face BERT implementation.
    """
    def __init__(self, opt, shared=None):
        # download pretrained models
        download(opt['datapath'])
        self.pretrained_path = os.path.join(opt['datapath'], 'models',
                                            'bert_models', MODEL_PATH)
        opt['pretrained_path'] = self.pretrained_path
        super().__init__(opt, shared)
        self.delimiter = opt['delimiter']

    @classmethod
    def history_class(cls):
        return BertClassifierHistory

    @staticmethod
    def add_cmdline_args(parser):
        TorchClassifierAgent.add_cmdline_args(parser)
        parser = parser.add_argument_group('BERT Classifier Arguments')
        parser.add_argument('--type-optimization', type=str,
                            default='all_encoder_layers',
                            choices=['additional_layers', 'top_layer',
                                     'top4_layers', 'all_encoder_layers',
                                     'all'],
                            help='which part of the encoders do we optimize '
                                 '(defaults to all layers)')
        parser.add_argument('--add-cls-token', type='bool', default=True,
                            help='add [CLS] token to text vec')
        parser.add_argument('--sep-last-utt', type='bool', default=False,
                            help='separate the last utterance into a different'
                                 'segment with [SEP] token in between')
        parser.set_defaults(
            dict_maxexs=0,  # skip building dictionary
        )

    @staticmethod
    def dictionary_class():
        return BertDictionaryAgent

    def build_model(self):
        num_classes = len(self.class_list)
        self.model = BertWrapper(
            BertModel.from_pretrained(self.pretrained_path),
            num_classes
        )

    def init_optim(self, params, optim_states=None, saved_optim_type=None):
        self.optimizer = get_bert_optimizer([self.model],
                                            self.opt['type_optimization'],
                                            self.opt['learningrate'])

    def _set_text_vec(self, *args, **kwargs):
        obs = super()._set_text_vec(*args, **kwargs)
        if self.opt.get('add_cls_token', True):
            # insert [CLS] token
            start_tensor = obs['text_vec'].new_tensor([self.dict.start_idx])
            obs['text_vec'] = torch.cat([start_tensor, obs['text_vec']], 0)
        return obs

    def score(self, batch):
        segment_idx = batch.text_vec * 0
        if self.opt.get('sep_last_utt', False):
            batch_len = batch.text_vec.size(1)
            # find where [SEP] token is
            seps = (batch.text_vec == self.dict.end_idx).nonzero()
            for row in seps:
                # set last utterance to segment 1
                segment_idx[row[0], list(range(row[1], batch_len))] = 1

        mask = (batch.text_vec != self.NULL_IDX).long()
        token_idx = batch.text_vec * mask
        return self.model(token_idx, segment_idx, mask)
