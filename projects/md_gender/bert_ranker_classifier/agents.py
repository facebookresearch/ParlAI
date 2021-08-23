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

    def train_step(self, batch):
        output = super().train_step(batch)
        preds = output['text']
        self._update_confusion_matrix(preds, batch.labels)
        return output

    def eval_step(self, batch):
        if batch.text_vec is None:
            return
        output = super().eval_step(batch)
        preds = output['text']
        labels = batch.labels
        if preds is not None and labels is not None:
            self._update_confusion_matrix(preds, labels)
        return output
