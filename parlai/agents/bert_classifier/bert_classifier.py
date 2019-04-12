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
from parlai.core.torch_classifier_agent import TorchClassifierAgent
from parlai.zoo.bert.build import download

import os
try:
    from pytorch_pretrained_bert import BertModel
except ImportError:
    raise Exception(("BERT rankers needs pytorch-pretrained-BERT installed. \n "
                     "pip install pytorch-pretrained-bert"))


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

    def score(self, batch):
        segment_idx = batch.text_vec * 0
        mask = (batch.text_vec != self.NULL_IDX).long()
        token_idx = batch.text_vec * mask
        return self.model(token_idx, segment_idx, mask)
