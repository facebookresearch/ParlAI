#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Imported from https://github.com/chaochun/nlu-asdiv-dataset
Paper: "A Diverse Corpus for Evaluating and Developing English Math Word Problem Solvers" (Miao et al 2021)
https://aclanthology.org/2020.acl-main.92.pdf
"""

import xml.etree.ElementTree as ET
import os

from .build import build
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from typing import Optional

from tasks.reasoning.agents import MWPStepsReasoningTeacher


class ASDivStepByStepReasoningTeacher(MWPStepsReasoningTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following math problem. Show your work.",
            thought_token="EQUATION: ",
        )
        return parser

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)  # NOTE: the call to build here
        opt['datafile'] = os.path.join(opt['datapath'], 'asdiv')
        self.id = 'asdiv'
        super().__init__(opt, shared)

    def load_data(self):
        data = []
        data_path = self.opt['datafile']
        tree = ET.parse(f"{data_path}/ASDiv.xml")
        root = tree.getroot()
        problemset_root = root.getchildren()[0]
        for problem in problemset_root.getchildren():
            problem_elements = problem.getchildren()
            question = problem_elements[0].text + " " + problem_elements[1].text
            answer = problem_elements[3].text
            equation = problem_elements[4].text
            problem_dict = {
                "problem": question,
                "solution": answer,
                "equation": equation,
            }
            data.append(problem_dict)
        return data

    def get_data_for_fold(self, fold):
        data = self.load_data()
        # Synthetically split train/test/valid splits.
        # ASDiv has 2305 problems.
        VALID_SAMPLES = 200
        TEST_SAMPLES = 300
        if "test" not in self.datatype:
            if "valid" in self.datatype:
                data = data[TEST_SAMPLES : TEST_SAMPLES + VALID_SAMPLES]
            elif "train" in self.datatype:
                data = data[TEST_SAMPLES + VALID_SAMPLES :]
            else:
                raise RuntimeError(f"Not a viable datatype {self.datatype}!")
        else:
            data = data[:TEST_SAMPLES]

        for problem in data:
            question = problem["problem"]
            answer = problem["solution"]
            steps = [problem['equation']]
            yield {"question": question, "answer": answer, "steps": steps}


class DefaultTeacher(ASDivStepByStepReasoningTeacher):
    pass
