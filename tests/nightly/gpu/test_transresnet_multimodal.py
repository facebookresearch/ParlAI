#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test for Transresnet Multimodal Pretrained Model.
"""

import unittest
import parlai.scripts.display_data as display_data
import parlai.utils.testing as testing_utils


MODEL_OPTIONS = {
    'task': 'image_chat:ImageChatTestTeacher',
    'model_file': 'models:image_chat/transresnet_multimodal/model',
    'datatype': 'test',
    'yfcc_path': 'temp',
}


@testing_utils.skipUnlessGPU
class TestTransresnet(unittest.TestCase):
    """
    Checks that pre-trained Transresnet Multimodal model gives the correct results.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up the test by downloading the model/data.
        """
        parser = display_data.setup_args()
        parser.set_defaults(**MODEL_OPTIONS)
        opt = parser.parse_args([], print_args=False)
        opt['num_examples'] = 1
        display_data.display_data(opt)

    def test_transresnet(self):
        """
        Test pretrained model.
        """
        _, test = testing_utils.eval_model(MODEL_OPTIONS, skip_valid=True)

        # Overall
        self.assertAlmostEqual(test['accuracy'], 0.3667, places=4)
        self.assertAlmostEqual(test['hits@5'], 0.6333, places=4)
        self.assertAlmostEqual(test['hits@10'], 0.7667, places=4)

        # First round
        self.assertAlmostEqual(test['first_round/hits@1/100'], 0.2, places=4)

        # Second round
        self.assertAlmostEqual(test['second_round/hits@1/100'], 0.5, places=4)

        # Third round
        self.assertAlmostEqual(test['third_round+/hits@1/100'], 0.4, places=4)


if __name__ == '__main__':
    unittest.main()
