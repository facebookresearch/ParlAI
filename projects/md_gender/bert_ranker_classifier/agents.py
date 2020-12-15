#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Using classification metrics for a ranking BERT model.
"""

from parlai.agents.bert_ranker.bi_encoder_ranker import BiEncoderRankerAgent
from projects.style_gen.modules import ClassificationMixin


class BertRankerClassifierAgent(ClassificationMixin, BiEncoderRankerAgent):
    """
    Bert BiEncoder that computes classification metrics (F1, precision, recall)
    """

    def get_labels_field(self, observations):
        if 'labels' in observations[0]:
            labels_field = 'labels'
        elif 'eval_labels' in observations[0]:
            labels_field = 'eval_labels'
        else:
            labels_field = None
        return labels_field

    def train_step(self, batch):
        output = super().train_step(batch)
        preds = output['text']
        labels_field = self.get_labels_field(batch['observations'])
        labels_lst = self._get_labels(batch['observations'], labels_field)
        self._update_confusion_matrix(preds, labels_lst)
        return output

    def eval_step(self, batch):
        output = super().eval_step(batch)
        preds = output['text']
        labels_field = self.get_labels_field(batch['observations'])
        if labels_field is not None:
            labels_lst = self._get_labels(batch['observations'], labels_field)
            self._update_confusion_matrix(preds, labels_lst)
        return output
