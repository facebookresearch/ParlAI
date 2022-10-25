#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import unittest
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils
from parlai.tasks.reasoning.question_answer import (
    MultipleChoiceQuestionAnswer,
)
from dataclasses import dataclass

Q_TOKEN = "QUESTION: "
Q = "question"
A_TOKEN = "ANSWER: "
A_CHOICES = "ANSWER_CHOICES: "


IN_SEP = "\t"  # same as default
EX_SEP = '\n'  # same as default


@dataclass
class TestCase:
    friendly_name: str
    flags: str
    question_text: str
    answer_text: str


TEST_CASES = [
    TestCase(
        "basic", "", f"QUESTION: question\tANSWER_CHOICES: (a) 1 (b) 2 (c) 3", "(b) 2"
    ),
    TestCase(
        "diff_opts_prefix",
        "--choices-prefix a b c d",
        f"QUESTION: question\tANSWER_CHOICES: a 1 b 2 c 3",
        "b 2",
    ),
]


@testing_utils.skipIfCircleCI
class TestMultipleChoiceQuestionAnswer(unittest.TestCase):
    def _run_case(self, case):
        parser = MultipleChoiceQuestionAnswer.add_cmdline_args(
            ParlaiParser(False, False), None
        )
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        multi_choice_qa = MultipleChoiceQuestionAnswer(opt)
        question_text, _, answer_text, _ = multi_choice_qa.get_question_answer(
            {"question": Q, "answer_choices": [1, 2, 3], "answer_idx": 1}
        )
        self.assertEqual(
            case.question_text,
            question_text,
            f"Question text different in {case.friendly_name}",
        )

        self.assertEqual(
            case.answer_text,
            answer_text,
            f"Answer text different in {case.friendly_name}",
        )

    def test_cases(self):
        for case in TEST_CASES:
            self._run_case(case)


if __name__ == '__main__':
    unittest.main()
