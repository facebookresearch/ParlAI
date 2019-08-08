#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test for Transresnet Multimodal Pretrained Model."""

import unittest
import parlai.scripts.display_data as display_data
import parlai.core.testing_utils as testing_utils


MODEL_OPTIONS = {
    'task': 'image_chat:ImageChatTestTeacher',
    'model_file': 'models:image_chat/transresnet_multimodal/model',
    'datatype': 'test',
    'yfcc_path': 'temp',
}


@testing_utils.skipUnlessGPU
class TestTransresnet(unittest.TestCase):
    """Checks that pre-trained Transresnet Multimodal model gives the correct results."""

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

        # Overall
        self.assertEqual(
            test['accuracy'],
            0.3667,
            'test accuracy = {}\nLOG:\n{}'.format(test['accuracy'], stdout),
        )
        self.assertEqual(
            test['hits@5'],
            0.633,
            'test hits@5 = {}\nLOG:\n{}'.format(test['hits@5'], stdout),
        )
        self.assertEqual(
            test['hits@10'],
            0.767,
            'test hits@10 = {}\nLOG:\n{}'.format(test['hits@10'], stdout),
        )

        # First round
        self.assertEqual(
            test['first_round']['hits@1/100'],
            0.2,
            'test first round hits@1/100 = {}\nLOG:\n{}'.format(
                test['first_round']['hits@1/100'], stdout
            ),
        )

        # Second round
        self.assertEqual(
            test['second_round']['hits@1/100'],
            0.5,
            'test second round hits@1/100 = {}\nLOG:\n{}'.format(
                test['second_round']['hits@1/100'], stdout
            ),
        )

        # Third round
        self.assertEqual(
            test['third_round+']['hits@1/100'],
            0.4,
            'test third round hits@1/100 = {}\nLOG:\n{}'.format(
                test['third_round+']['hits@1/100'], stdout
            ),
        )


if __name__ == '__main__':
    unittest.main()
