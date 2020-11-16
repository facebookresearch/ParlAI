#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy
import torch.nn.functional as F

from parlai.core.worlds import DialogPartnerWorld, validate
import parlai.tasks.md_gender.utils as gend_utils


def get_threshold_preds(
    act, axis, self_threshold=-1, partner_threshold=-1, prettify=True
):
    pred = act['text']
    if prettify:
        pred = pred.split(':')[-1]

    if 'sorted_scores' in act:
        scores = F.softmax(act['sorted_scores'].float()).tolist()
        scores = [round(x, 4) for x in scores]
        cands = act['text_candidates']
        if prettify:
            cands = [
                cand.split(':')[-1].replace('gender-neutral', 'neutral')
                for cand in cands
            ]

        if axis == 'self' and self_threshold > 0:
            if scores[0] > self_threshold:
                pred = cands[0]
            else:
                pred = gend_utils.UNKNOWN
        elif axis == 'partner' and partner_threshold > 0:
            if cands[0] == 'neutral':
                pred = cands[0]
            else:  # male or female
                if scores[0] > partner_threshold:
                    pred = cands[0]
                else:
                    pred = gend_utils.UNKNOWN

        if prettify:
            strs = [f'{cand} ({score})' for cand, score in zip(cands, scores)]
            pred += '\n\t' + '[ ' + '\t'.join(strs) + ']'

    return pred


def get_axis_predictions(
    act_label, model_agent, self_threshold, partner_threshold, return_acts=False
):
    act = deepcopy(act_label)
    if 'labels' in act:
        del act['labels']
    if 'eval_labels' in act:
        del act['eval_labels']

    # SELF (as) pred
    act.force_set('label_candidates', gend_utils.SELF_CANDS)
    model_agent.observe(validate(act))
    self_act = model_agent.act()
    if not return_acts:
        self_pred = get_threshold_preds(
            self_act,
            axis='self',
            self_threshold=self_threshold,
            partner_threshold=partner_threshold,
        )
    else:
        self_pred = self_act

    # PARTNER (to) pred
    act.force_set('label_candidates', gend_utils.PARTNER_CANDS)
    model_agent.observe(validate(act))
    partner_act = model_agent.act()
    if not return_acts:
        partner_pred = get_threshold_preds(
            partner_act,
            axis='partner',
            self_threshold=self_threshold,
            partner_threshold=partner_threshold,
        )
    else:
        partner_pred = partner_act

    # ABOUT pred
    act.force_set('label_candidates', gend_utils.ABOUT_CANDS)
    model_agent.observe(validate(act))
    about_act = model_agent.act()
    if not return_acts:
        about_pred = get_threshold_preds(
            about_act,
            axis='about',
            self_threshold=self_threshold,
            partner_threshold=partner_threshold,
        )
    else:
        about_pred = about_act

    return self_pred, partner_pred, about_pred


class InteractiveWorld(DialogPartnerWorld):
    """
    Interact with a model and get TO/AS/ABOUT predictions for each utterance.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group('Gender Multiclass Interactive World')
        parser.add_argument(
            '--self-threshold',
            type=float,
            default=0.52,
            help='Threshold for choosing unknown for self',
        )
        parser.add_argument(
            '--partner-threshold',
            type=float,
            default=0.52,
            help='Threshold for choosing unknown for self',
        )
        argparser.set_params(
            single_turn=True,  # this is a single turn task currently
            eval_candidates='inline',
            return_cand_scores=True,
        )

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        acts = self.acts
        human_agent, model_agent = self.agents

        # human act
        act = deepcopy(human_agent.act())

        self_pred, partner_pred, about_pred = get_axis_predictions(
            act,
            model_agent,
            self_threshold=self.opt['self_threshold'],
            partner_threshold=self.opt['partner_threshold'],
        )

        pred_text = f'SELF: {self_pred}\nPARTNER: {partner_pred}\nABOUT: {about_pred}'
        acts[1] = {'id': 'MDGender Classifier', 'text': pred_text, 'episode_done': True}

        human_agent.observe(validate(acts[1]))
        self.update_counters()
