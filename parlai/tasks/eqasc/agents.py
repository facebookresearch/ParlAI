#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Imported from https://github.com/harsh19/Reasoning-Chains-MultihopQA
Paper: "Learning to Explain: Datasets and Models for Identifying Valid Reasoning Chains in Multihop Question-Answering"
(Jhamtani and Clark 2020) https://aclanthology.org/2020.emnlp-main.10.pdf

Multihop Explanations for QASC dataset.
"""

import json
import os

from parlai.utils.data import DatatypeHelper

from .build import build
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.io import PathManager
from typing import Optional

from tasks.reasoning.agents import StepByStepReasoningTeacher
import random


class EqascStepByStepReasoningTeacher(StepByStepReasoningTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following common sence problem. Show your work."
        )
        parser.add_argument(
            "--extrinsic-step",
            type=bool,
            default=False,
        )
        return parser

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        opt['datafile'] = os.path.join(opt['datapath'], 'eqasc')
        self.id = 'eqasc'
        self.fold = DatatypeHelper.fold(opt['datatype'])
        self.eqasc_random = random.Random(42)
        super().__init__(opt, shared)

    def get_data_for_fold(self, fold):
        extrinsic_step = self.opt.get("extrinsic_step", False)
        suffix = self.fold
        if self.fold == "valid":
            suffix = "dev"
        filename = os.path.join(self.opt['datafile'], f"eqasc_{suffix}_grc.json")

        json_data = json.load(PathManager.open(filename))

        if len(json_data) == 0:
            return

        data = []
        for example in json_data:
            question = example['question']['stem'].strip()
            f1 = example['fact1']
            f2 = example['fact2']
            steps = [f1, f2]
            answer = example['combinedfact']
            m = {
                "question": question,
                "answer": answer,
                "steps": steps,
            }
            if not extrinsic_step:
                yield m
            else:
                data.append(m)

        if extrinsic_step:
            for line in data:
                rand_steps = self.eqasc_random.choice(data)["steps"]
                random_step = self.eqasc_random.choice(rand_steps)
                yield {
                    "question": line["question"],
                    "answer": line["answer"],
                    "steps": line["steps"],
                    "extrinsic_step": random_step,
                }


class DefaultTeacher(EqascStepByStepReasoningTeacher):
    pass
