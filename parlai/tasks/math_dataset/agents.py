#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Imported from https://github.com/hendrycks/math/
Paper: "Measuring Mathematical Problem Solving With the MATH Dataset" (Hendricks et al 2021) https://arxiv.org/pdf/2103.03874.pdf

"""

import glob
import json
import os
import random
import re

from parlai.tasks.math_dataset.build import build
from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.utils.io import PathManager
from typing import List, Optional

from parlai.tasks.reasoning.agents import MWPStepsReasoningTeacher


DOMAINS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


class MathDatasetStepByStepReasoningTeacher(MWPStepsReasoningTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following math problem. Show your work."
        )
        parser.add_argument(
            "--math_dataset-domains",
            nargs="+",
            default=DOMAINS,
            choices=DOMAINS,
        )
        parser.add_argument(
            "--math-latex-converter",
            type=bool,
            default=False,
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
        suffix = 'test' if opt['datatype'].startswith('test') else 'train'
        opt['datafile'] = os.path.join(opt['datapath'], 'MATH', suffix)
        self.id = 'math_dataset'
        self.math_random = random.Random(42)
        super().__init__(opt, shared)

    def load_data(self, domains) -> List[str]:
        data = []
        data_path = self.opt['datafile']
        for domain in domains:
            for filename in glob.glob(f"{data_path}/{domain}/*.json", recursive=True):
                with PathManager.open(filename, "r") as f:
                    example = json.load(f)
                    data.append(example)
        return data

    def get_data_for_fold(self, fold):
        domains = self.opt.get("math_dataset_domains", DOMAINS)
        convert = self.opt.get("math_latex_converter", False)
        extrinsic_step = self.opt.get("extrinsic_step", False)
        data = self.load_data(domains)

        if "test" not in self.datatype:
            # Synthetically split some of train out so we get a valid set.
            # MATHS has 7.5k train, 5k test, 12.5 total.
            split_size = len(data) // 10
            if "valid" in self.datatype:
                data = data[:split_size]
            elif "train" in self.datatype:
                data = data[split_size:]
            else:
                raise RuntimeError(f"Not a viable datatype {self.datatype}!")

        for problem in data:
            question = problem["problem"]
            answer_blob = problem["solution"]
            final_answer = self.get_boxed_answer(answer_blob)
            answer_blob = self._clean_steps(answer_blob)
            steps = answer_blob.split(". ")
            if extrinsic_step:
                random_step = self._find_nonempty_random_step(data)
            if convert:
                question = self._latex_conversion(question)
                final_answer = self._latex_conversion(final_answer)
                final_steps = []
                for step in steps:
                    final_steps.append(self._latex_conversion(step))
                if extrinsic_step:
                    random_step = self._latex_conversion(random_step)
            else:
                final_steps = steps
            while len(final_steps) > 0 and (
                final_steps[-1] == '' or final_steps[-1] == ' '
            ):
                final_steps = final_steps[:-1]
            if extrinsic_step:
                yield {
                    "question": question,
                    "answer": final_answer,
                    "steps": final_steps,
                    "extrinsic_step": random_step,
                }
                continue
            yield {"question": question, "answer": final_answer, "steps": final_steps}

    def _clean_steps(self, steps: str) -> str:
        steps = steps.replace("\n\n", "\n")
        steps = steps.replace("$$", "$ ")
        steps = steps.replace("\n", ". ")
        steps = steps.replace(".$", "$. ")
        steps = steps.replace("..", ".")
        steps = steps.replace(",$", "$,")
        steps = steps.replace(". . ", ". ")

        return steps

    def _latex_conversion(self, final_answer: str) -> str:
        """
        Following page 18 at https://arxiv.org/pdf/2206.14858.pdf with some
        modifications.
        """
        substitutions = [
            ('an ', ''),  # ('a ', ''),
            ('.$', '$'),
            ('\\$', ''),
            (r'\ ', ''),
            # (' ', ''),
            ('  ', ' '),
            ('mbox', 'text'),
            (',\\text{and}', ','),
            ('\\text{and}', ','),
            ('\\text{m}', '\\text{}'),
            ('\\cdot', '*'),
            ('\\left', ''),
            ('\\right', ''),
        ]
        removed_expressions = [
            # 'square', 'ways', 'integers', 'dollars', 'mph', 'inches', 'ft',
            # 'hours', 'km', 'units',
            '\\ldots',
            # 'sue', 'points', 'feet',
            # 'minutes', 'digits', 'cents', 'degrees', 'cm', 'gm', 'pounds',
            # 'meters', 'meals', 'edges', 'students', 'childrentickets', 'multiples',
            '\\text{s}',
            '\\text{.}',
            '\\text{\ns}',
            '\\text{}^2',
            '\\text{}^3',
            '\\text{\n}',
            '\\text{}',
            r'\mathrm{th}',
            r'^\circ',
            r'^{\circ}',
            r'\;',
            r',\!',
            '{,}',
            '"',
            '\\dots',
        ]

        # final_answer = final_answer.split('=')[-1]
        for before, after in substitutions:
            final_answer = final_answer.replace(before, after)
        for expr in removed_expressions:
            final_answer = final_answer.replace(expr, '')

        # Extract answer that is in LaTeX math, is bold,
        # is surrounded by a box, etc.
        # final_answer = re.sub(r'(.*?)(\$)(.*?)(\$)(.*)', '$\\3$', final_answer)
        final_answer = re.sub(r'(\\text\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\textbf\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\overline\{)(.*?)(\})', '\\2', final_answer)
        final_answer = re.sub(r'(\\boxed\{)(.*)(\})', '\\2', final_answer)

        # Normalize shorthand TeX:
        # \fracab -> \frac{a}{b}
        # \frac{abc}{bef} -> \frac{abc}{bef}
        # \fracabc -> \frac{a}{b}c
        # \sqrta -> \sqrt{a}
        # \sqrtab -> sqrt{a}b
        final_answer = re.sub(r'(frac)([^{])(.)', 'frac{\\2}{\\3}', final_answer)
        final_answer = re.sub(r'(sqrt)([^{])', 'sqrt{\\2}', final_answer)
        final_answer = final_answer.replace('$', '')
        # replace \frac{a}{b}
        final_answer = re.sub(
            r'(\\frac){(-?[0-9]*[a-x]*)}{(-?[0-9]*[a-x]*)}', '\\2/\\3', final_answer
        )
        # replace \frac{a}b
        final_answer = re.sub(r'(\\frac){(-?[0-9a-x^()]*)}', '\\2/', final_answer)

        # Normalize 100,000 -> 100000
        if final_answer.replace(',', '').isdigit():
            final_answer = final_answer.replace(',', '')

        return final_answer

    def _find_nonempty_random_step(self, dataset: List[str]) -> str:
        '''Here we *ASSUME* that the whole dataset contains at least one non-empty step
        Otherwise it will go into infinite loop looking for the one
        '''
        # what we call an empty step
        empty_steps = ["", " "]
        # first find chain with at least one non-empty step
        rand_steps = self._clean_steps(
            self.math_random.choice(dataset)["solution"]
        ).split(". ")
        # make sure this chain has at least one non-empty step
        i = 0
        while i < len(rand_steps) and rand_steps[i] in empty_steps:
            i += 1
        # if it doesn't, try again
        if i == len(rand_steps):
            return self._find_nonempty_random_step(dataset)
        random_step = empty_steps[0]
        # find non-empty random step (and we know it exists in this chain)
        while random_step in empty_steps:
            random_step = self.math_random.choice(rand_steps)
        return random_step

    def get_boxed_answer(self, answer):
        boxed_idx = answer.find("boxed{")
        final_answer = answer[boxed_idx:]
        final_answer = final_answer[
            final_answer.find("{") + 1 : final_answer.rfind("}")
        ]
        final_answer = self.clean_answer(final_answer)
        return final_answer

    def clean_answer(self, answer):
        while answer.rfind("}") < answer.rfind('{'):
            answer = answer[: answer.rfind("}")]
        return answer


class DefaultTeacher(MathDatasetStepByStepReasoningTeacher):
    pass
