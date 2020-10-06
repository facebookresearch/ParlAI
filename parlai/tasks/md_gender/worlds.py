#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from copy import deepcopy

from parlai.core.message import Message
from parlai.core.worlds import create_task
from parlai.core.worlds import DialogPartnerWorld, validate
from parlai.agents.repeat_label.repeat_label import RepeatLabelAgent

import random
from typing import List

import torch.nn.functional as F

from parlai.tasks.md_gender.utils import (
    SELF_CANDS,
    PARTNER_CANDS,
    ABOUT_CANDS,
    UNKNOWN,
    get_control_token,
)


def get_threshold_preds(
    act, axis, self_threshold=-1, partner_threshold=-2, prettify=True
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
                pred = UNKNOWN
        elif axis == 'partner' and partner_threshold > 0:
            if cands[0] == 'neutral':
                pred = cands[0]
            else:  # male or female
                if scores[0] > partner_threshold:
                    pred = cands[0]
                else:
                    pred = UNKNOWN

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
    act.force_set('label_candidates', SELF_CANDS)
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
    act.force_set('label_candidates', PARTNER_CANDS)
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
    act.force_set('label_candidates', ABOUT_CANDS)
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
            default=0.53,
            help='Threshold for choosing unknown for self',
        )
        parser.add_argument(
            '--partner-threshold',
            type=float,
            default=0.55,
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
        acts[1] = {'text': pred_text, 'episode_done': True}

        human_agent.observe(validate(acts[1]))
        self.update_counters()


CONTROL_TOK_LST = [
    ('female', 'female', 'female'),
    ('male', 'male', 'male'),
    ('unknown', 'unknown', 'gender-neutral'),
    ('female', 'male', 'gender-neutral'),
    ('male', 'female', 'gender-neutral'),
]

GEN_SAMPLES = [
    'hi!',
    'hello!',
    'How are you?',
    'What are you up to?',
    'what is your job?',
    'what do you do for fun?',
    'tell me about yourself',
    'what is your name?',
    'where are you?',
    'tell me a joke!',
]


class InteractiveGeneratorWorld(DialogPartnerWorld):
    """
    Interactive world for controllable generative model.
    """

    @staticmethod
    def add_cmdline_args(argparser):
        parser = argparser.add_argument_group(
            'Gender Multiclass Interactive Generator World'
        )
        parser.add_argument(
            '--automatic-appends',
            type='bool',
            default=True,
            help='Automatically append various tokens; otherwise, get as input',
        )
        parser.add_argument(
            '--get-gen-samples',
            type='bool',
            default=False,
            help='Get generation samples instead of ',
        )
        parser.add_argument(
            '--controllable',
            type='bool',
            default=False,
            help='Append controllable generation tokens',
        )
        argparser.set_params(single_turn=True)

    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        self.controllable = self.opt['controllable']
        self.automatic_appends = self.opt['automatic_appends']
        self.get_gen_samples = self.opt['get_gen_samples']
        self.idx = 0

    def parley(self):
        """
        Agent 0 goes first.

        Alternate between the two agents.
        """
        acts = self.acts
        human_agent, model_agent = self.agents

        if not self.get_gen_samples or self.idx >= len(GEN_SAMPLES):
            act = deepcopy(human_agent.act())
        else:
            print('\n\n' + '~' * 50)
            print(
                f'Generating response for message:\t{GEN_SAMPLES[self.idx]}\n'
                + '~' * 50
            )
            act = Message({'text': GEN_SAMPLES[self.idx], 'episode_done': True})
            self.idx += 1

        if self.controllable:
            if not self.automatic_appends:
                preds = []
                while len(preds) < 3:
                    human_agent.observe(
                        {
                            'id': 'system',
                            'text': 'Enter control tokens for SELF/PARTNER/ABOUT (M/N/F/U):\t',
                        }
                    )
                    tok_act = human_agent.act()
                    preds = [x for x in tok_act['text'].split(' ')[:3] if x]

                mp = {'m': 'male', 'n': 'gender-neutral', 'f': 'female', 'u': 'unknown'}
                preds = [mp[pred.lower()] for pred in preds]
                control_tok = get_control_token(preds[0], preds[1], preds[2])
                print(f'\tPrepending control token: {control_tok}')
                act.force_set('text', act['text'] + ' ' + control_tok)
                model_agent.observe(act)
                acts[1] = model_agent.act()
                human_agent.observe(validate(acts[1]))
            else:
                if self.controllable:
                    act_orig_text = act['text']
                    for tok in CONTROL_TOK_LST:
                        control_tok = get_control_token(tok[0], tok[1], tok[2])
                        print(f'\tPrepending control token: {control_tok}')
                        act.force_set('text', act_orig_text + ' ' + control_tok)
                        model_agent.observe(act)
                        acts[1] = model_agent.act()
                        human_agent.observe(validate(acts[1]))
        else:
            # not controllable
            model_agent.observe(act)
            acts[1] = model_agent.act()
            human_agent.observe(validate(acts[1]))

        self.update_counters()
