#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Tests for AbstractReasoningTeacher.
"""
import unittest
from parlai.core.params import ParlaiParser

try:
    import checklist  # noqa
    from parlai.tasks.math_dataset.agents import (
        MathDatasetStepByStepReasoningTeacher,
    )

    DEPENDENCIES_AVAILABLE = True
except ImportError:
    DEPENDENCIES_AVAILABLE = False

from dataclasses import dataclass


@dataclass
class TestCase:
    name: str
    dirty_answer: str
    clean_answer: str


@unittest.skipUnless(DEPENDENCIES_AVAILABLE, 'Must install checklist to run this test')
class TestMathDatasetStepByStepReasoningTeacher(unittest.TestCase):
    def test_get_boxed_answer(self):
        TEST_CASES = [
            TestCase(
                "Closed bracket before open bracket",
                "sum_{p = 1981}^{2007} 45 \\\\\n&= 2 \\sum_{k = 1}^{44} k^2 + 27 \\cdot 45 \\\\\n&= 2 \\cdot \
                    \\frac{44 \\cdot 45 \\cdot 89}{6} + 27 \\cdot 45 \\\\\n&= \\boxed{59955}.\n\\end{align*}",
                "59955",
            ),
            TestCase("Common case", "boxed{59955}", "59955"),
        ]
        parser = MathDatasetStepByStepReasoningTeacher.add_cmdline_args(
            ParlaiParser(False, False), None
        )
        opt, _ = parser.parse_and_process_known_args()
        opt["datatype"] = 'train'  # arbitrary
        teacher = MathDatasetStepByStepReasoningTeacher(opt)

        for case in TEST_CASES:
            clean_answer = teacher.get_boxed_answer(case.dirty_answer)
            self.assertEqual(
                clean_answer, case.clean_answer, f"Text different in {case.name}"
            )


if __name__ == '__main__':
    unittest.main()
