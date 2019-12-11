#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test Teachers.

A module for testing various teacher types in ParlAI
"""

import os
import unittest
from parlai.utils import testing as testing_utils
import regex as re


class TestAbstractImageTeacher(unittest.TestCase):
    """
    Test AbstractImageTeacher.
    """

    def test_display_data(self):
        """
        Test that, with pre-loaded image features, all examples are different.
        """

        def _test_display_output(opt):
            output = testing_utils.display_data(opt)
            train_labels = re.findall(r"\[labels: .*\]", output[0])
            valid_labels = re.findall(r"\[eval_labels: .*\]", output[1])
            test_labels = re.findall(r"\[eval_labels: .*\]", output[2])

            for i, lbls in enumerate([train_labels, valid_labels, test_labels]):
                self.assertGreater(len(lbls), 0, 'DisplayData failed')
                self.assertEqual(len(lbls), len(set(lbls)), output[i])

        with testing_utils.tempdir() as tmpdir:
            data_path = tmpdir
            os.makedirs(os.path.join(data_path, 'ImageTeacher'))

            opt = {'task': 'integration_tests:ImageTeacher', 'datapath': data_path}
            for image_mode in ['resnet152', 'no_image_model']:
                opt['image_mode'] = image_mode
                _test_display_output(opt)


if __name__ == '__main__':
    unittest.main()
