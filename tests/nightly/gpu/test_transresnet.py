#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test for Transresnet Pretrained Model."""

import unittest
import parlai.scripts.display_data as display_data
import parlai.core.testing_utils as testing_utils


MODEL_OPTIONS = {
    'task': 'personality_captions:PersonalityCaptionsTestTeacher',
    'model_file': 'models:personality_captions/transresnet/model',
    'datatype': 'test',
    'yfcc_path': 'temp',
    'num_test_labels': 5
}


@testing_utils.skipUnlessGPU
class TestTransresnet(unittest.TestCase):
    """Checks that pre-trained Transresnet model gives the correct results."""

    @classmethod
    def setUpClass(cls):
        """Set up the test by downloading the model/data."""
        with testing_utils.capture_output():
            parser = display_data.setup_args()
            parser.set_defaults(**MODEL_OPTIONS)
            opt = parser.parse_args(print_args=False)
            opt['num_examples'] = 1
            display_data.display_data(opt)

    def test_transresnet(self):
        """Test pretrained model."""
        stdout, _, test = testing_utils.eval_model(MODEL_OPTIONS, skip_valid=True)
        self.assertEqual(
            test['accuracy'], 0.4,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout)
        )
        self.assertEqual(
            test['hits@5'], 0.9,
            'test hits@5 = {}\nLOG:\n{}'.format(test['hits@5'], stdout)
        )
        self.assertEqual(
            test['hits@10'], 0.9,
            'test hits@10 = {}\nLOG:\n{}'.format(test['hits@10'], stdout)
        )
        self.assertEqual(
            test['med_rank'], 2.0,
            'test med_rank = {}\nLOG:\n{}'.format(test['med_rank'], stdout)
        )


if __name__ == '__main__':
    unittest.main()
