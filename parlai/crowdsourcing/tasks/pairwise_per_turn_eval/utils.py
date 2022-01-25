#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from parlai.crowdsourcing.utils.tests import AbstractParlAIChatTest

import unittest
from typing import Any, Dict, Optional, Tuple


class AbstractPerTurnEvalTest(AbstractParlAIChatTest, unittest.TestCase):
    """
    Abstract test class for testing per-turn eval code.
    """

    # TODO: de-duplicate or refactor all these methods!

    def _check_output_key(self, key: str, actual_value: Any, expected_value: Any):
        """
        Special logic for handling the 'final_chat_data' key.
        """
        if key == 'final_chat_data':
            self._check_final_chat_data(
                actual_value=actual_value, expected_value=expected_value
            )
        else:
            super()._check_output_key(
                key=key, actual_value=actual_value, expected_value=expected_value
            )

    def _check_final_chat_data(
        self, actual_value: Dict[str, Any], expected_value: Dict[str, Any]
    ):
        """
        Check the actual and expected values of the final chat data.
        """
        for key_inner, expected_value_inner in expected_value.items():
            if key_inner == 'dialog':
                assert len(actual_value[key_inner]) == len(expected_value_inner)
                for actual_message, expected_message in zip(
                    actual_value[key_inner], expected_value_inner
                ):
                    self.assertEqual(
                        {k: v for k, v in actual_message.items() if k != 'message_id'},
                        {
                            k: v
                            for k, v in expected_message.items()
                            if k != 'message_id'
                        },
                    )
            elif key_inner == 'task_description':
                for (key_inner2, expected_value_inner2) in expected_value_inner.items():
                    if key_inner2 == 'model_file':
                        pass
                        # The path to the model file depends on the random
                        # tmpdir
                    elif key_inner2 == 'model_opt':
                        keys_to_ignore = [
                            'datapath',
                            'dict_file',
                            'model_file',
                            'override',
                            'parlai_home',
                            'starttime',
                        ]
                        # These paths depend on the random tmpdir and the host
                        # machine
                        for (
                            key_inner3,
                            expected_value_inner3,
                        ) in expected_value_inner2.items():
                            if key_inner3 in keys_to_ignore:
                                pass
                            else:
                                self.assertEqual(
                                    actual_value[key_inner][key_inner2][key_inner3],
                                    expected_value_inner3,
                                    f'Error in key {key_inner3}!',
                                )
                    else:
                        self.assertEqual(
                            actual_value[key_inner][key_inner2],
                            expected_value_inner2,
                            f'Error in key {key_inner2}!',
                        )
            else:
                self.assertEqual(
                    actual_value[key_inner],
                    expected_value_inner,
                    f'Error in key {key_inner}!',
                )
