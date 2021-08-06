#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from projects.md_gender.bert_ranker_classifier.agents import BertRankerClassifierAgent
from parlai.agents.bert_classifier.bert_classifier import BertClassifierAgent
from parlai.core.params import ParlaiParser
from typing import Optional
from parlai.core.opt import Opt

from parlai.core.metrics import SumMetric
from parlai.tasks.md_gender.utils import ALL_CANDS

EVAL_LABEL_GROUPS = {'gender:' + cand: ALL_CANDS[cand] for cand in ALL_CANDS}


class SubgroupSaverAgentTrait(object):
    """
    resets the label_candidates and eval_labels at the observe stage to
    self.label_candidates and self.eval_labels.
    """

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        grp = super().add_cmdline_args(parser, partial_opt=partial_opt)
        grp.add_argument(
            '--eval_labels',
            default=['gender:about'],
            type=str,
            nargs='*',
            help=f"The possible evaluation labels or name for the evaluation label group (ie. gender:about)",
        )
        grp.add_argument(
            '--show-counts',
            default=False,
            type='bool',
            help='whether or not to show counts of each subgroup (for batches of 1)',
        )
        return parser

    def __init__(self, opt, shared=None) -> None:
        super().__init__(opt, shared)
        eval_labels = opt['eval_labels']
        if len(opt['eval_labels']) == 1:
            eval_labels = EVAL_LABEL_GROUPS[opt['eval_labels'][0]]
        self.label_candidates = eval_labels
        self.eval_labels = eval_labels
        self.show_counts = opt.get('show_counts', False)

    def observe(self, observation):
        if hasattr(self, 'label_candidates'):
            observation.force_set('label_candidates', self.label_candidates)
        if hasattr(self, 'eval_labels'):
            observation.force_set('old_labels', observation.get('eval_labels', []))
            observation.force_set('eval_labels', self.eval_labels)
        observation.force_set('episode_done', True)
        observation = super().observe(observation)
        return observation

    def act(self, last_label_only=False):
        output = super().act()
        # keep track of counts instead
        if self.show_counts:
            metrics_dic = output['metrics']
            labels = output['text']
            if isinstance(output['text'], str):
                labels = [output['text']]
            for txt in labels:
                key = 'total_cnt_predicted_' + txt
                if key in metrics_dic:
                    metrics_dic[key] += SumMetric(1)
                else:
                    metrics_dic[key] = SumMetric(1)

        # only keep the part of the label after `:`
        if last_label_only:
            new_text = [label.split(':')[-1] for label in labels]
            if len(new_text) == 1:
                new_text = new_text[0]
            output.force_set('text', new_text)
        return output


class SubgroupBertRankerClassifier(SubgroupSaverAgentTrait, BertRankerClassifierAgent):
    """
    Example usage:

    -m parlai_internal.projects.model_cards_subgroup.agents:SubgroupBertRankerClassifier
    """

    pass


class SubgroupBertClassifier(SubgroupSaverAgentTrait, BertClassifierAgent):
    """
    Example usage:

    -m parlai_internal.projects.model_cards_subgroup.agents:SubgroupBertClassifier
    """

    pass
