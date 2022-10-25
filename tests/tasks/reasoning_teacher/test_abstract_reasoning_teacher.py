#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils
from parlai.tasks.reasoning.base import (
    AbstractQuestionAnswer,
    AbstractReason,
    AbstractReasoningTeacher,
)
from dataclasses import dataclass

Q = "question"
A_TOKEN = "ANSWER: "
A = "answertext"
R_TOKEN = "REASON: "
R = "reasontext"
EXEMPLAR_SUFFIX = "exemplar"

IN_SEP = "\t"  # same as default
EX_SEP = '\n'  # same as default

EXEMPLAR_STRING = f"{Q}{EXEMPLAR_SUFFIX}{IN_SEP}{R_TOKEN}{R}{EXEMPLAR_SUFFIX}{IN_SEP}{A_TOKEN}{A}{EXEMPLAR_SUFFIX}"

TASK_PROMPT = "prompt"
P = TASK_PROMPT + EX_SEP


@dataclass
class TestCase:
    friendly_name: str
    flags: str
    text: str
    label: str


TEST_CASES = [
    TestCase("default", "", f"{P}{Q}{IN_SEP}{R_TOKEN}", f"{R}{IN_SEP}{A_TOKEN}{A}"),
    TestCase(
        "default__gen_answer",
        "--initial-generation-target answer",
        f"{P}{Q}{IN_SEP}{R_TOKEN}{R}{IN_SEP}{A_TOKEN}",
        f"{A}",
    ),
    TestCase(
        "no_reason_simple", "--include-reason False", f"{P}{Q}{IN_SEP}{A_TOKEN}", f"{A}"
    ),
    TestCase(
        "no_reason_token_in_label",
        "--include-reason False --include-generation-target-token-in-input False",
        f"{P}{Q}{IN_SEP}",
        f"{A_TOKEN}{A}",
    ),
    TestCase(
        "reason_token_in_label__gen_reason",
        "--include-generation-target-token-in-input False",
        f"{P}{Q}{IN_SEP}",
        f"{R_TOKEN}{R}{IN_SEP}{A_TOKEN}{A}",
    ),
    TestCase(
        "answer_before_reason__gen_reason",
        "--reason-before-answer False",
        f"{P}{Q}{IN_SEP}{A_TOKEN}{A}{IN_SEP}{R_TOKEN}",
        f"{R}",
    ),
    TestCase(
        "answer_before_reason_token_in_label__gen_reason",
        "--reason-before-answer False --include-generation-target-token-in-input False",
        f"{P}{Q}{IN_SEP}{A_TOKEN}{A}{IN_SEP}",
        f"{R_TOKEN}{R}",
    ),
    TestCase(
        "reason_token_in_label__gen_answer",
        "--initial-generation-target answer --include-generation-target-token-in-input False",
        f"{P}{Q}{IN_SEP}{R_TOKEN}{R}{IN_SEP}",
        f"{A_TOKEN}{A}",
    ),
    TestCase(
        "answer_before_reason__gen_answer",
        "--initial-generation-target answer --reason-before-answer False",
        f"{P}{Q}{IN_SEP}{A_TOKEN}",
        f"{A}{IN_SEP}{R_TOKEN}{R}",
    ),
    TestCase(
        "answer_before_reason_token_in_label__gen_answer",
        "--initial-generation-target answer --reason-before-answer False --include-generation-target-token-in-input False",
        f"{P}{Q}{IN_SEP}",
        f"{A_TOKEN}{A}{IN_SEP}{R_TOKEN}{R}",
    ),
    TestCase(
        "exemplar",
        "--exemplar-idx 0",
        f"{P}{EXEMPLAR_STRING}{EX_SEP}{Q}{IN_SEP}{R_TOKEN}",
        f"{R}{IN_SEP}{A_TOKEN}{A}",
    ),
]


class DummyQuestionAnswer(AbstractQuestionAnswer):
    def get_question_answer(self, example_dict):
        return (example_dict[Q], A_TOKEN, example_dict[A], {})


class DummyReason(AbstractReason):
    def get_full_reason_text(self, example_dict):
        return (R_TOKEN, example_dict[R], {})


class DummyAbstractReasoningTeacher(AbstractReasoningTeacher):
    @classmethod
    def add_cmdline_args(cls, parser, partial_opt):
        parser = super().add_cmdline_args(parser, partial_opt)
        parser.set_defaults(task_prompt=TASK_PROMPT)
        return parser

    @classmethod
    def get_reason_class(self) -> AbstractReason:
        return DummyReason

    @classmethod
    def get_question_answer_class(self) -> AbstractQuestionAnswer:
        return DummyQuestionAnswer

    def get_data_for_fold(self, fold):
        yield {Q: Q, A: A, R: R}

    def __init__(self, opt, shared=None):
        self.exemplars_raw = [
            {Q: Q + EXEMPLAR_SUFFIX, A: A + EXEMPLAR_SUFFIX, R: R + EXEMPLAR_SUFFIX}
        ]
        super().__init__(opt, shared)


@testing_utils.skipIfCircleCI
class TestAbstractReasoningTeacher(unittest.TestCase):
    def _run_case(self, case):
        parser = DummyAbstractReasoningTeacher.add_cmdline_args(
            ParlaiParser(False, False), None
        )
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        opt["datatype"] = 'train'  # arbitrary to make things happy
        opt["datafile"] = 'train'  # arbitrary to make things happy
        teacher = DummyAbstractReasoningTeacher(opt)
        for example, _ in teacher.setup_data("dummy_fold"):
            print(example)
            print(case.text)
            self.assertEqual(
                case.text, example["text"], f"Text different in {case.friendly_name}"
            )
            self.assertEqual(
                case.label, example["label"], f"Label different in {case.friendly_name}"
            )

    def test_cases(self):
        for case in TEST_CASES:
            self._run_case(case)


if __name__ == '__main__':
    unittest.main()
