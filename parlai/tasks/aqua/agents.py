#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import random
from typing import Optional

from parlai.tasks.reasoning.agents import MWPStepsReasoningTeacher
from parlai.tasks.reasoning.base import (
    AbstractQuestionAnswer,
    AbstractReason,
    AbstractReasoningTeacher,
)
from parlai.tasks.reasoning.question_answer import MultipleChoiceQuestionAnswer
from parlai.tasks.reasoning.reason_types.step_by_step import StepByStepReason

from parlai.core.opt import Opt
from parlai.core.params import ParlaiParser
from parlai.core.teachers import DialogTeacher
from parlai.utils.io import PathManager
from parlai.tasks.aqua.build import build

import os
import copy
import json


AQUA = 'AQuA'
AQUA_QUESTION_KEY = 'question'
AQUA_ANSWER_KEY = 'correct'
AQUA_OPTIONS_KEY = 'options'
AQUA_RATIONALE_KEY = 'rationale'
RATIONALE_QUESTION_TEXT = 'Can you provide a rationale for your answer?'


def _path(opt):
    build(opt)

    dt = opt['datatype'].split(':')[0]

    if dt == 'train':
        prefix = 'train'
    # Using matched set as valid and mismatched set as test
    elif dt == 'valid':
        prefix = 'dev'
    elif dt == 'test':
        prefix = 'test'
    else:
        raise RuntimeError('Not valid datatype.')

    data_path = os.path.join(opt['datapath'], AQUA, AQUA, prefix + '.tok.json')

    return data_path


def setup_data(path):
    print('loading: ' + path)

    with PathManager.open(path, 'r') as data_file:
        for line in data_file:
            question = json.loads(line)
            question_text = question[AQUA_QUESTION_KEY]
            answer = ord(question[AQUA_ANSWER_KEY]) - ord('A')
            labels = question[AQUA_OPTIONS_KEY]
            answer = [labels[answer]]
            yield (question_text, answer, None, labels), True

            # Ask for a rationale now
            rationale = [question[AQUA_RATIONALE_KEY]]
            yield (RATIONALE_QUESTION_TEXT, rationale, None, rationale), False


class DefaultTeacher(DialogTeacher):
    def __init__(self, opt, shared=None):
        opt = copy.deepcopy(opt)
        data_path = _path(opt)
        opt['datafile'] = data_path
        self.id = 'AQuA'

        super().__init__(opt, shared)

    def setup_data(self, path):
        return setup_data(path)


class AQuAReasoningTeacher(AbstractReasoningTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following math problem. Show your work.",
            choices_prefix=["", "", "", "", ""],
        )
        return parser

    @classmethod
    def get_reason_class(self) -> AbstractReason:
        return StepByStepReason

    @classmethod
    def get_question_answer_class(self) -> AbstractQuestionAnswer:
        return MultipleChoiceQuestionAnswer

    def __init__(self, opt, shared=None):
        self.datatype = opt['datatype']
        build(opt)
        opt['datafile'] = _path(opt)
        self.id = 'aqua'
        super().__init__(opt, shared)

    def get_data_for_fold(self, fold):
        with PathManager.open(self.opt['datafile']) as f:
            data = f.readlines()

        for line in data:
            problem = json.loads(line)
            question = problem["question"]
            answer = problem["correct"]
            #  A -> 0; B -> 1; C -> 2; D -> 3; E -> 4
            answer_idx = ord(answer) - 65
            options = problem["options"]
            rationale = problem["rationale"]
            steps = rationale.split("\n")
            yield {
                "question": question,
                "steps": steps,
                "answer_idx": answer_idx,
                "answer_choices": options,
            }


class AQuAStepByStepReasoningTeacher(MWPStepsReasoningTeacher):
    @classmethod
    def add_cmdline_args(
        cls, parser: ParlaiParser, partial_opt: Optional[Opt] = None
    ) -> ParlaiParser:
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(
            task_prompt="Solve the following math problem. Show your work.",
            choices_prefix=["", "", "", "", ""],
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
        opt['datafile'] = _path(opt)
        self.id = 'aqua'
        self.aqua_random = random.Random(42)
        super().__init__(opt, shared)

    def get_data_for_fold(self, fold):
        extrinsic_step = self.opt.get("extrinsic_step", False)

        with PathManager.open(self.opt['datafile']) as f:
            data = f.readlines()

        messages = []
        for line in data:
            problem = json.loads(line)
            question = problem["question"].replace(' .', '.').replace(' ,', ',')
            options = {}
            for x in problem["options"]:
                try:
                    a, b = x.split(' ) ')
                except ValueError:
                    if '(' not in x:
                        # some lines have typos and duplicate indices
                        a, c, b = x.split(' ) ')
                    else:
                        # example: B ) ( 4 ! - 1 ) * ( 5 ! - 1 ) * ( 6 ! - 1 )
                        a = x.split(' ) ')[0]
                        b = " ) ".join(x.split(' ) ')[1:])
                options[a] = b
            answer = options[problem["correct"]]
            rationale = problem["rationale"]
            steps = rationale.split("\n")
            final_steps = []
            # some steps are just dots or empty
            for step in steps:
                while len(step) > 0 and (step[-1] == '.' or step[-1] == ' '):
                    step = step[:-1]
                if len(step) > 0:
                    final_steps.append(step)
            m = {"question": question, "steps": final_steps, "answer": answer}
            if not extrinsic_step:
                yield m
            else:
                messages.append(m)

        if extrinsic_step:
            for line in messages:
                rand_steps = self.aqua_random.choice(messages)["steps"]
                random_step = self.aqua_random.choice(rand_steps)
                yield {
                    "question": line["question"],
                    "answer": line["answer"],
                    "steps": line["steps"],
                    "extrinsic_step": random_step,
                }
