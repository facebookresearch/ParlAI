#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

try:
    import apex  # noqa: F401

    APEX_AVAILABLE = True
except ImportError:
    APEX_AVAILABLE = False


import unittest
from parlai.core.agents import create_agent
from parlai.core.params import ParlaiParser
import parlai.utils.testing as testing_utils


@unittest.skipIf(APEX_AVAILABLE, "Apex is installed, can't test its absence.")
class TestNoApex(unittest.TestCase):
    """
    Test if some models that were pretrained with APEX.

    They should load on CPU and GPU, even if the user doesn't have apex installed.
    """

    def test_import(self):
        with self.assertRaises(ImportError):
            import apex as _  # noqa: F401

    def test_fused_adam(self):
        with self.assertRaises(ImportError):
            # we should crash if the user tries not giving --opt adam
            testing_utils.train_model(
                dict(
                    model_file='zoo:unittest/apex_fused_adam/model',
                    task='integration_tests:nocandidate',
                )
            )
        # no problem if we give the option

        pp = ParlaiParser(True, True)
        opt = pp.parse_args(
            [
                '--model-file',
                'zoo:unittest/apex_fused_adam/model',
                '--dict-file',
                'zoo:unittest/apex_fused_adam/model.dict',
                '--task',
                'integration_tests:nocandidate',
                '--optimizer',
                'adam',
            ]
        )
        create_agent(opt, requireModelExists=True)

    def test_fp16(self):
        # nice clean fallback if no fp16
        valid, test = testing_utils.eval_model(
            dict(
                model_file='zoo:unittest/apex_fp16/model',
                task='integration_tests:nocandidate',
                num_examples=4,
            )
        )
        assert valid['accuracy'] == 1.0
        assert test['accuracy'] == 1.0

        # also no problem if we explicitly turn it on
        valid, test = testing_utils.eval_model(
            dict(
                model_file='zoo:unittest/apex_fp16/model',
                task='integration_tests:nocandidate',
                num_examples=4,
                fp16=True,
            )
        )
        assert valid['accuracy'] == 1.0
        assert test['accuracy'] == 1.0

        with self.assertRaises(RuntimeError):
            # we will have some fp16 tokens missing if we turn of fp16
            # note: this test could be made unnecessary in the future if we improve
            # the fp16 logic
            valid, test = testing_utils.eval_model(
                dict(
                    model_file='zoo:unittest/apex_fp16/model',
                    task='integration_tests:nocandidate',
                    num_examples=4,
                    fp16=False,
                )
            )

        valid, test = testing_utils.eval_model(
            dict(
                model_file='zoo:unittest/apex_fp16/model',
                task='integration_tests:nocandidate',
                num_examples=4,
                force_fp16_tokens=False,
            )
        )
        assert valid['accuracy'] == 1.0
        assert test['accuracy'] == 1.0
