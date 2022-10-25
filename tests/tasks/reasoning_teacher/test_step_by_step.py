#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for AbstractReasoningTeacher.
"""
from typing import List
import unittest
from parlai.core.params import ParlaiParser

try:
    import checklist  # noqa
    from parlai.tasks.reasoning.reason_types.step_by_step import (
        StepByStepReason,
        MWPStepsReason,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from dataclasses import dataclass

INPUT_BASE_STEPS = ['a', 'b', 'c', 'd']
EXTRINSIC_STEP = 'e'
INPUT_MATH_STEPS = ['(2t + 1)(-2)/7 = 123456789*0', '3t = 12t / 4']
INPUT_TEXT_STEPS = [
    'John is Irish because he was born in Ireland.',
    'Mark Stewart was born and raised in Chicago.',
    'Rates will continue to rise.',
    '42',
]


SP_FLAG = "--step-perturbations "
MWP_SP_FLAG = "--math-step-perturbations "


@dataclass
class TestCase:
    friendly_name: str
    flags: str
    steps_string: str


@dataclass
class TextTestCase:
    friendly_name: str
    flags: str
    input_steps: List[str]
    steps_string: str


BASE_TEST_CASES = [  # empirically determined when random used
    TestCase("default", "", "abcd"),
    TestCase("shuffle", SP_FLAG + "ShuffleSteps", "adbc"),
    TestCase("dup-one", SP_FLAG + "DuplicateOneStep", "aabcd"),
    TestCase("rem-one", SP_FLAG + "RemoveOneStep", "bcd"),
    TestCase("swap-one", SP_FLAG + "SwapOneStep", "dbca"),
    TestCase("extrinsic-step", SP_FLAG + "ExtrinsicHallucinatedStep", "eabcd"),
    # Test that order respected of pertubations
    TestCase("shuffle_dup-one", SP_FLAG + "ShuffleSteps DuplicateOneStep", "aadbc"),
    TestCase("dup-one_shuffle", SP_FLAG + "DuplicateOneStep ShuffleSteps", "adbac"),
]

BASE_TEST_CASES_EMPTY_STEPS = [
    TestCase("dup-one-empty", SP_FLAG + "DuplicateOneStep", ""),
    TestCase("rem-one-empty", SP_FLAG + "RemoveOneStep", ""),
    TestCase("swap-one-empty", SP_FLAG + "SwapOneStep", ""),
    TestCase("extrinsic-step-empty", SP_FLAG + "ExtrinsicHallucinatedStep", ""),
]

MATH_TEST_CASES = [  # empirically determined when random used
    TestCase(
        "shuffle-num", MWP_SP_FLAG + "ShuffleNumbers", "(0t + 2)(-123456789)/2 = 1*7"
    ),
    TestCase(
        "shuffle-ops", MWP_SP_FLAG + "ShuffleOperations", "(2t + 1)(*2)-7 = 123456789/0"
    ),
    TestCase("rand-num", MWP_SP_FLAG + "RandomNumber", "(81t + 1)(-2)/7 = 123456789*0"),
    TestCase(
        "rand-op", MWP_SP_FLAG + "RandomOperation", "(2t + 1)(+2)/7 = 123456789*0"
    ),
]

CHAINED_BASE_MATH_TEST_CASES = [  # empirically determined when random used
    TestCase(
        "shuffle-num-shuffle-step",
        SP_FLAG + "SwapOneStep " + MWP_SP_FLAG + "ShuffleNumbers",
        "4t = 3t / 12\n(0t + 2)(-1)/2 = 123456789*7",
    ),
    TestCase(
        "shuffle-num-shuffle-step",
        MWP_SP_FLAG + "ShuffleNumbers " + SP_FLAG + "RemoveOneStep",
        "4t = 3t / 12",
    ),
]

TEXT_TEST_CASES = [  # empirically determined when random used
    TextTestCase(
        "negate",
        SP_FLAG + "NegateStep",
        [
            'John is Irish because he was born in Ireland.',
            'Mark Stewart was born and raised in Chicago.',
            'Rates will continue to rise.',
            '42',
        ],
        "John is Irish because he was born in Ireland. Mark Stewart was born and raised in Chicago. Rates won't continue to rise. 42",
    ),
    TextTestCase(
        "grammar error drop verb",
        SP_FLAG + "GrammaticalErrorStep",
        [
            'John is Irish because he was born in Ireland.',
            'Mark Stewart was born and raised in Chicago.',
            'Rates will continue to rise.',
            '42',
        ],
        "John is Irish because he was in Ireland. Mark Stewart was born and raised in Chicago. Rates will continue to rise. 42",
    ),
    TextTestCase(
        "grammar error swap word",
        SP_FLAG + "GrammaticalErrorStep",
        [
            'John is Irish because he was born in Ireland.',
        ],
        "is John Irish because he was born in Ireland.",
    ),
    TextTestCase(
        "grammar error tense change",
        SP_FLAG + "GrammaticalErrorStep",
        [
            'John is Irish because he was born in Ireland.',
            'Mark Stewart was born and raised in Chicago.',
            'John was',
            'John was',
            'John was',
        ],
        "John is Irish because he was born in Ireland. Mark Stewart be bear and raise in Chicago. John was John was John was",
    ),
]


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, 'Must install checklist to run this test')
class TestStepPertubations(unittest.TestCase):
    def _run_case(self, case, iteration):
        parser = StepByStepReason.add_cmdline_args(ParlaiParser(False, False), None)
        parser.set_defaults(
            thought_token="", inner_separator_token=""
        )  # So it's cleaner and there's no separators
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        steps_reason = StepByStepReason(opt)
        _, steps, _ = steps_reason.get_full_reason_text(
            {"steps": INPUT_BASE_STEPS, "extrinsic_step": EXTRINSIC_STEP}
        )
        self.assertEqual(
            case.steps_string,
            steps,
            f"Text different in {case.friendly_name}; iteration {iteration}",
        )

    def _run_case_empty(self, case):
        parser = StepByStepReason.add_cmdline_args(ParlaiParser(False, False), None)
        parser.set_defaults(thought_token="", inner_separator_token="")
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        steps_reason = StepByStepReason(opt)
        with self.assertRaises(RuntimeError):
            steps_reason.get_full_reason_text({"steps": []})

    def _run_case_missing_key(self):
        case = TestCase(
            "extrinsic-step-missing-key", SP_FLAG + "ExtrinsicHallucinatedStep", ""
        )
        parser = StepByStepReason.add_cmdline_args(ParlaiParser(False, False), None)
        parser.set_defaults(thought_token="", inner_separator_token="")
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        steps_reason = StepByStepReason(opt)
        with self.assertRaises(RuntimeError):
            steps_reason.get_full_reason_text({"steps": ['a']})

    def run_mwp_case(self, case):
        parser = MWPStepsReason.add_cmdline_args(ParlaiParser(False, False), None)
        parser.set_defaults(thought_token="")
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        steps_reason = MWPStepsReason(opt)
        _, steps, _ = steps_reason.get_full_reason_text(
            {"steps": [INPUT_MATH_STEPS[0]]}
        )
        self.assertEqual(
            case.steps_string,
            steps,
        )

    def run_chained_case(self, case):
        parser = MWPStepsReason.add_cmdline_args(ParlaiParser(False, False), None)
        parser.set_defaults(thought_token="", inner_separator_token="\n")
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        steps_reason = MWPStepsReason(opt)
        _, steps, _ = steps_reason.get_full_reason_text({"steps": INPUT_MATH_STEPS})
        self.assertEqual(
            case.steps_string,
            steps,
        )

    def run_text_case(self, case):
        parser = StepByStepReason.add_cmdline_args(ParlaiParser(False, False), None)
        parser.set_defaults(thought_token="", inner_separator_token=" ")
        opt = parser.parse_args(case.flags.split(" ") if len(case.flags) > 0 else [])
        steps_reason = StepByStepReason(opt)
        _, steps, _ = steps_reason.get_full_reason_text({"steps": case.input_steps})
        self.assertEqual(
            case.steps_string,
            steps,
            f"Text different in {case.friendly_name}",
        )

    def test_cases(self):
        for case in BASE_TEST_CASES:
            self._run_case(case, 1)
            # Run each twice to make sure we've got the random seeds set
            self._run_case(case, 2)
        for case in BASE_TEST_CASES_EMPTY_STEPS:
            self._run_case_empty(case)
        for case in MATH_TEST_CASES:
            self.run_mwp_case(case)
        for case in CHAINED_BASE_MATH_TEST_CASES:
            self.run_chained_case(case)
        for case in TEXT_TEST_CASES:
            self.run_text_case(case)


if __name__ == '__main__':
    unittest.main()
