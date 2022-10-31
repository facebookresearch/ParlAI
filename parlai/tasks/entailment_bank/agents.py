#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Imported from https://allenai.org/data/entailmentbank
Paper: "Explaining Answers with Entailment Trees" (Dalvi et al 2021) https://aclanthology.org/2021.emnlp-main.585.pdf

2k multi-step entailment trees, explaining the answers to ARC science questions.
"""

import json
import os
import re

from parlai.utils.data import DatatypeHelper

from .build import build
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.io import PathManager
from typing import List, Optional

from tasks.reasoning.agents import StepByStepReasoningTeacher
import random


class EntailmentBankStepByStepReasoningTeacher(StepByStepReasoningTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following entailment problem. Show your work."
        )
        parser.add_argument(
            "--task-id",
            type=str,
            default="1",
            choices=["1", "2", "3"],
            help="Use data from Task 1, 2, or 3 as defined in the paper",
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
        opt['datafile'] = os.path.join(opt['datapath'], 'entailment_bank')
        self.id = 'entailment_bank'
        self.fold = DatatypeHelper.fold(opt['datatype'])
        self.entailment_random = random.Random(42)
        super().__init__(opt, shared)

    def get_data_for_fold(self, fold):
        extrinsic_step = self.opt.get("extrinsic_step", False)
        task_id = self.opt.get("task_id", "1")
        suffix = self.fold
        if self.fold == "valid":
            suffix = "dev"
        filename = os.path.join(
            self.opt['datafile'],
            f"entailment_trees_emnlp2021_data_v3/dataset/task_{task_id}/{suffix}.jsonl",
        )

        data = []
        with PathManager.open(filename) as f:
            for x in f.readlines():
                example = json.loads(x)
                context = example['context']
                context = self._parse_context(context)
                # question with context
                question = context + ". " + example['question']
                proof = example['full_text_proof']
                steps = self._get_steps(proof)
                answer = example['answer']
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
                rand_steps = self.entailment_random.choice(data)["steps"]
                random_step = self.entailment_random.choice(rand_steps)
                yield {
                    "question": line["question"],
                    "answer": line["answer"],
                    "steps": line["steps"],
                    "extrinsic_step": random_step,
                }

    def _parse_context(self, cont: str) -> str:
        """
        Change format.

        Input: sent1: xxx sent2: yyy sent3: zzz
        Output: xxx. yyy. zzz.
        """
        parsed = re.split(r" ?sent\d+: ", cont)[1:]
        return ". ".join(parsed)

    def _get_steps(self, proof: str) -> List[str]:
        """
        Input:
            proof: " [BECAUSE] earth is a kind of celestial object [AND] a star is a kind of celestial
        object / celestial body [AND] apparent motion is when an object appears to move relative to
        another object 's position [INFER] int1: apparent motion of stars is when stars appear to move
        relative to earth's position [BECAUSE] int1 [AND] the earth rotating on its axis causes stars
        to appear to move across the sky at night [INFER] int2: the earth rotating on its axis causes
        apparent motion of stars [BECAUSE] int2 [AND] stars appear to move relative to the horizon
        during the night [INFER] int3: the earth rotating on its axis causes stars to move relative to
        the horizon during the night"
        Return:
            ['earth is a kind of celestial object', 'a star is a kind of celestial object / celestial
            body', "apparent motion is when an object appears to move relative to another object 's
            position", "Therefore apparent motion of stars is when stars appear to move relative to
            earth's position", 'the earth rotating on its axis causes stars to appear to move across
            the sky at night', 'Therefore the earth rotating on its axis causes apparent motion of
            stars', 'stars appear to move relative to the horizon during the night', 'Therefore the
            earth rotating on its axis causes stars to move relative to the horizon during the night']
        """
        steps = re.split(
            r" \[BECAUSE\] int\d+ \[AND\] | \[BECAUSE\] | \[AND\] ", proof
        )[1:]
        out = []
        for step in steps:
            sub_step = re.sub(r"int\d+\:", "Therefore", step)
            out.extend(sub_step.split(" [INFER] "))
        return out


class DefaultTeacher(EntailmentBankStepByStepReasoningTeacher):
    pass
