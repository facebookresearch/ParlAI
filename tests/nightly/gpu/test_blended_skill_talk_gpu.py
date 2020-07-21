#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from parlai.utils import testing as testing_utils


class TestBlendedSkillTalkModels(unittest.TestCase):
    def test_accuracies(self):
        """
        Test accuracies of BST models in the zoo.
        """
        model_names_to_desired_accuracies = {
            'bst_single_task': 0.8906,
            'convai2_single_task': 0.8438,
            'ed_single_task': 0.7656,
            'wizard_single_task': 0.7500,
            'multi_task': 0.9062,
            'multi_task_bst_tuned': 0.9219,
        }
        for model_name, desired_accuracy in model_names_to_desired_accuracies.items():
            valid, _ = testing_utils.eval_model(
                opt={
                    'batchsize': 16,
                    'model_file': f'zoo:blended_skill_talk/{model_name}/model',
                    'task': 'blended_skill_talk',
                    'num_examples': 64,
                },
                skip_test=True,
            )
            self.assertAlmostEqual(valid['accuracy'], desired_accuracy, delta=0.005)


if __name__ == '__main__':
    unittest.main()
