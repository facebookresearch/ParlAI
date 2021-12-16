#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
LIGHT RPA Unlikelihood Agent.

Utilizes a left-to-right RPA Classifier to predict tokens that yield incorrect character
classifications.
"""
from typing import Optional, List, Tuple
from parlai.core.params import ParlaiParser
from parlai.core.opt import Opt
from parlai.core.torch_agent import Batch
import torch
import torch.nn
import torch.nn.functional as F

from parlai.agents.transformer.transformer import TransformerGeneratorAgent
from parlai.core.torch_agent import Output
from parlai.core.metrics import AverageMetric

from projects.light_whoami.agents.rpa_rerank import RPAReranker
from projects.msc.agents.long_tga import TransformerVariantAgent


def div(x, y):
    if y == 0:
        return 0
    else:
        return x / y


class RpaUlAgent(TransformerGeneratorAgent):
    """
    RPA UL Agent.

    Performs unlikelihood such that tokens which lead to misclassification are
    penalized.
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.pred_logsoftmax = torch.nn.LogSoftmax(dim=2)
        self.character_softmax = F.softmax
        self.top_k = opt['ul_top_k_toks']
        self.only_wrong_class_toks = opt['only_wrong_class_toks']
        self.all_wrong_class_toks = opt['all_wrong_class_toks']
        self.random_toks = opt['random_toks']
        self.initial_classification_threshold = opt['initial_classification_threshold']
        if shared:
            self.classifier = shared['classifier']
        else:
            self.classifier = RPAReranker(opt)
        # set debug to get the full context.
        # we can just as easily decode the full_text_vec, but figured
        # UL is already quite slow so i dont think background preprocessing
        # will yield much speedup
        self.is_debug = True

    def share(self):
        shared = super().share()
        shared['classifier'] = self.classifier
        return shared

    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        super().add_cmdline_args(parser, partial_opt=partial_opt)  # type: ignore
        RPAReranker.add_cmdline_args(parser, partial_opt)
        grp = parser.add_argument_group('RPA UL Group')
        grp.add_argument(
            '--rpa-ul-ratio',
            default=0.5,
            type=float,
            help='how often to apply unlikelihood per train step',
        )
        grp.add_argument(
            '--ul-top-k-toks',
            type=int,
            default=1,
            help='how many tokens to penalize with unlikelihood loss',
        )
        grp.add_argument(
            '--only-wrong-class-toks',
            type='bool',
            default=False,
            help='Whether to only apply UL to tokens that result in wrong character classification',
        )
        grp.add_argument(
            '--all-wrong-class-toks',
            type='bool',
            default=False,
            help='If True, apply UL to ALL tokens in an utterance that result in wrong character classification',
        )
        grp.add_argument(
            '--random-toks',
            type='bool',
            default=False,
            help='If True, apply UL to random tokens in an utterance that results in wrong character classification',
        )
        grp.add_argument(
            '--initial-classification-threshold',
            default=0.5,
            type=float,
            help='level of confidence for the classifier in order to apply UL. '
            'I.e., wrong character score must be greater than this, or else example is skipped',
        )
        grp.add_argument('--eval-skip-generation', default=True, type="bool")
        return parser

    def compute_loss(self, batch, return_output=False):
        """
        Perform several steps to include unlikelihood loss.

        1. Generate from model
        2. Compute a forward pass to score generations
        3. Determine which tokens to apply UL to
        4. Compute the UL loss
        5. Sum the losses, record metrics.
        """
        total_loss, model_output = super().compute_loss(batch, return_output=True)

        if (
            self.is_training and (torch.rand(1).item() >= self.opt['rpa_ul_ratio'])
        ) or (not self.is_training and self.opt['eval_skip_generation']):
            if return_output:
                return total_loss, model_output
            else:
                return total_loss

        # Generate
        maxlen = self.label_truncate or 256
        with torch.no_grad():
            beam_pred_scores, _ = self._generate(batch, self.beam_size, maxlen)

        # forward pass to create graph for beam search case
        generations = [g[1:] for (g, _, _) in beam_pred_scores]
        pred_toks = torch.nn.utils.rnn.pad_sequence(generations, batch_first=True)
        model_output = self.model(*self._model_input(batch), ys=pred_toks)
        logits, *_ = model_output

        # construct mask marking incorrectly classified characters
        label_mask = torch.zeros_like(pred_toks).type_as(logits)
        (
            label_mask,
            wrong_class_cnt,
            wrong_class_all_cnt,
            right_class_cnt,
        ) = self.compute_ul_label_mask(label_mask, generations, batch)
        # Compute unlikelihood loss
        ul_loss = self.compute_ul_loss(pred_toks, label_mask, logits)  # type: ignore
        if label_mask.sum() > 0:
            total_loss += div(ul_loss.sum(), label_mask.sum())
        else:
            total_loss += ul_loss.sum()

        # record metrics
        self.record_local_metric(
            'ul_loss', AverageMetric.many(ul_loss.sum(dim=-1), label_mask.sum(dim=-1))
        )
        self.record_local_metric('wrong_class', AverageMetric.many(wrong_class_cnt))
        self.record_local_metric(
            'wrong_class_all', AverageMetric.many(wrong_class_all_cnt)
        )
        self.record_local_metric('right_class', AverageMetric.many(right_class_cnt))

        if return_output:
            return total_loss, model_output
        return total_loss

    def compute_ul_loss(
        self,
        pred_toks: torch.LongTensor,
        label_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Comptue UL Loss.

        :param pred_toks:
            predicted tokens from model
        :param label_mask:
            tokens to apply loss to
        :logits:
            token scores from model

        :return ul_loss:
            return the UL loss.
        """
        clamp_min = 1e-6 if self.opt['fp16'] else 1e-20
        lprobs = self.pred_logsoftmax(logits)
        pred_lprobs = lprobs.view(-1, lprobs.size(2)).gather(1, pred_toks.view(-1, 1))
        one_minus_probs = torch.clamp((1.0 - pred_lprobs.exp()), min=clamp_min).view(
            pred_toks.size(0), pred_toks.size(1)
        )
        loss = -torch.log(one_minus_probs) * label_mask
        return loss

    def compute_ul_label_mask(
        self,
        label_mask: torch.Tensor,
        generations: List[torch.LongTensor],
        batch: Batch,
    ) -> Tuple[torch.Tensor, List[int], List[int], List[int]]:
        """
        Compute Label Mask for UL.

        We classify the generations from a model in a left-to-right fashion.

        The topk tokens for which the character classification drops the most
        are chosen to apply UL loss.

        :param label_mask:
            label mask to fill
        :param generations:
            generations from the model
        :param batch:
            batch being considered

        :return (label_mask, wrong_cnt, wrong_cnt_below_threshold, right_cnt):
            label_mask: mask indicating which tokens to apply UL
            wrong_cnt: how many generations were classified as the wrong character (above a certain threshold)
            wrong_class_all_cnt: how many *total* generations were classified as wrong character
            right_cnt: how many generations were classified as the right character
        """
        wrong_class_cnt = []
        wrong_class_all_cnt = []
        right_class_cnt = []

        for i, gen in enumerate(generations):
            gen_i = gen.tolist()

            obs_i = batch.observations[i]
            context = obs_i['full_text']

            # first, classify full response
            me, _you = self.classifier.get_predictor_label_candidates(obs_i, context)
            full_classification = self.classifier.classify(context, self._v2t(gen_i))
            if full_classification['text'] == me:
                right_class_cnt.append(1)
                wrong_class_cnt.append(0)
                wrong_class_all_cnt.append(0)
                continue
            else:
                scores = self.character_softmax(
                    full_classification['sorted_scores'].float(), dim=0
                )
                if scores[0] > self.initial_classification_threshold:
                    right_class_cnt.append(0)
                    wrong_class_cnt.append(1)
                    wrong_class_all_cnt.append(1)
                else:
                    right_class_cnt.append(0)
                    wrong_class_cnt.append(0)
                    wrong_class_all_cnt.append(1)
                    continue

            # classify tokens left to right
            classifications = self.classifier.batch_classify(
                [context] * (len(gen_i) - 1),
                [self._v2t(gen_i[:end_idx]) for end_idx in range(1, len(gen_i))],
            )
            preds = []
            scores = []
            me_scores = torch.zeros(len(classifications))
            for j, c in enumerate(classifications):
                pred_c = c['text']
                scores_c = self.character_softmax(c['sorted_scores'].float(), dim=0)
                preds.append(pred_c)
                scores.append(scores_c)
                me_scores[j] = scores_c[0] if pred_c == me else scores_c[1]

            if self.only_wrong_class_toks:
                # we normalize all correct classifications here to be 1.0.
                # therefore, it is necessarily the case that the topk diffs
                # will only be tokens that yield wrong classification
                me_mask = me_scores > 0.5
                me_scores[me_mask] = 1.0
            me_scores_shift = torch.cat([me_scores[0:1], me_scores[:-1]])
            # for each token t_{i}, subtract prob t_{i-1} to see difference in
            # probability. if this is positive, it means the correct character probability
            # **decreased**. Thus, we take the topk
            if self.all_wrong_class_toks:
                assert self.only_wrong_class_toks
                top_offenders = (
                    (me_scores_shift - me_scores).greater(0).nonzero().squeeze()
                )
            elif self.random_toks:
                top_offenders = torch.randint(me_scores.size(0), (self.top_k,))
            else:
                top_offenders = (me_scores_shift - me_scores).topk(self.top_k).indices
            label_mask[i, top_offenders] = 1

        return label_mask, wrong_class_cnt, wrong_class_all_cnt, right_class_cnt

    def eval_step(self, batch):
        """
        Evaluate a single batch of examples.
        """
        if batch.text_vec is None:
            return
        self.model.eval()

        if batch.label_vec is not None:
            # calculate loss on targets with teacher forcing
            loss = self.compute_loss(batch)  # noqa: F841  we need the side effects

        maxlen = self.label_truncate or 256
        beam_preds_scores, _ = self._generate(batch, self.beam_size, maxlen)
        preds, scores, _ = zip(*beam_preds_scores)

        cand_choices = None
        text = [self._v2t(p) for p in preds] if preds is not None else None
        return Output(text, cand_choices)


class LongRpaUlAgent(TransformerVariantAgent, RpaUlAgent):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        RpaUlAgent.add_cmdline_args(parser, partial_opt)
        TransformerVariantAgent.add_cmdline_args(parser, partial_opt)
        return parser
