#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test components of specific crowdsourcing tasks.
"""

import unittest
from parlai.crowdsourcing.utils.acceptability import AcceptabilityChecker


class TestAcceptabilityChecker(unittest.TestCase):
    """
    Test the base AcceptabilityChecker class.
    """

    def test_sample_inputs(self):
        """
        Test sample inputs/outputs for the acceptability checker.
        """

        # Define test cases
        test_cases = [
            {  # Should pass
                'messages': [
                    'Hi - how are you?',
                    'What? Whatever for?',
                    'Wow, that sounds like a lot of work.',
                    "No, I don't expect he would be too happy about that either.",
                    "I don't even know where you would find that many squirrels.",
                    'Well, let me know if you need an extra hand.',
                ],
                'is_worker_0': False,
                'expected_violations': '',
            },
            {
                'messages': ['Hi', 'What?', 'Wow', "No", "I don't even know", 'Well,'],
                'is_worker_0': False,
                'expected_violations': 'under_min_length',
            },
            {  # Should fail, because the first worker shouldn't start with a greeting
                'messages': [
                    'Hi - how are you?',
                    'What? Whatever for?',
                    'Wow, that sounds like a lot of work.',
                    "No, I don't expect he would be too happy about that either.",
                    "I don't even know where you would find that many squirrels.",
                    'Well, let me know if you need an extra hand.',
                ],
                'is_worker_0': True,
                'expected_violations': 'starts_with_greeting',
            },
            {
                'messages': [
                    'HEYYYYYYY',
                    'What? Whatever for?',
                    'Wow, that sounds like a lot of work.',
                    "No, I don't expect he would be too happy about that either.",
                    "I don't even know where you would find that many squirrels.",
                    'WELLLLL LEMME KNOOOOOO',
                ],
                'is_worker_0': False,
                'expected_violations': 'too_much_all_caps',
            },
            {
                'messages': [
                    'Hi - how are you?',
                    'What? Whatever for?',
                    'Wow, that sounds like a lot of work.',
                    "No, I don't expect he would be too happy about that either.",
                    "I don't even know where you would find that many squirrels.",
                    'Hi - how are you?',
                ],
                'is_worker_0': False,
                'expected_violations': 'exact_match',
            },
            {
                'messages': [
                    'Hi - how are you?',
                    'What? Whatever for?',
                    'Wow, that sounds like a lot of work.',
                    "No, I don't expect he would be too happy about that either.",
                    "I don't even know where you would find that many squirrels.",
                    'Well, let me know if you need an extra hand.',
                    "I'm gonna say something that's totally XXX!",
                ],
                'is_worker_0': False,
                'expected_violations': 'unsafe:7',
            },
        ]
        test_cases_with_errors = [
            {
                'messages': ['Message 1', 'Message 2'],
                'is_worker_0': True,
                'violation_types': ['non_existent_violation_type'],
                'expected_exception': ValueError,
            }
        ]

        # Create checker
        acceptability_checker = AcceptabilityChecker()

        # Run through violation test cases
        for test_case in test_cases:
            actual_violations = acceptability_checker.check_messages(
                messages=test_case['messages'],
                is_worker_0=test_case['is_worker_0'],
                violation_types=acceptability_checker.possible_violation_types,
            )
            self.assertEqual(actual_violations, test_case['expected_violations'])

        # Run through test cases that should raise an error
        for test_case in test_cases_with_errors:
            with self.assertRaises(test_case['expected_exception']):
                acceptability_checker.check_messages(
                    messages=test_case['messages'],
                    is_worker_0=test_case['is_worker_0'],
                    violation_types=test_case['violation_types'],
                )


if __name__ == "__main__":
    unittest.main()
