#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from parlai.utils import testing as testing_utils


SHARED_OPTS = {'batchsize': 16, 'task': 'blended_skill_talk', 'num_examples': 64}


class TestBlendedSkillTalkModels(unittest.TestCase):
    """
    Test accuracies of BST models in the zoo.
    """

    def test_bst_single_task(self):
        """
        Test model trained single-task on BlendedSkillTalk.
        """
        valid, _ = testing_utils.eval_model(
            opt={
                **SHARED_OPTS,
                'model_file': f'zoo:blended_skill_talk/bst_single_task/model',
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['accuracy'], 0.8906, delta=0.005)

    def test_convai2_single_task(self):
        """
        Test model trained single-task on ConvAI2.
        """
        valid, _ = testing_utils.eval_model(
            opt={
                **SHARED_OPTS,
                'model_file': f'zoo:blended_skill_talk/convai2_single_task/model',
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['accuracy'], 0.8438, delta=0.005)

    def test_ed_single_task(self):
        """
        Test model trained single-task on EmpatheticDialogues.
        """
        valid, _ = testing_utils.eval_model(
            opt={
                **SHARED_OPTS,
                'model_file': f'zoo:blended_skill_talk/ed_single_task/model',
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['accuracy'], 0.7656, delta=0.005)

    def test_wizard_single_task(self):
        """
        Test model trained single-task on Wizard of Wikipedia.
        """
        valid, _ = testing_utils.eval_model(
            opt={
                **SHARED_OPTS,
                'model_file': f'zoo:blended_skill_talk/wizard_single_task/model',
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['accuracy'], 0.7500, delta=0.005)

    def test_multi_task(self):
        """
        Test model trained multi-task on dialogue datasets.
        """
        valid, _ = testing_utils.eval_model(
            opt={
                **SHARED_OPTS,
                'model_file': f'zoo:blended_skill_talk/multi_task/model',
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['accuracy'], 0.9062, delta=0.005)

    def test_multi_task_bst_tuned(self):
        """
        Test model trained multi-task and then tuned on BlendedSkillTalk.
        """
        valid, _ = testing_utils.eval_model(
            opt={
                **SHARED_OPTS,
                'model_file': f'zoo:blended_skill_talk/multi_task_bst_tuned/model',
            },
            skip_test=True,
        )
        self.assertAlmostEqual(valid['accuracy'], 0.9219, delta=0.005)


if __name__ == '__main__':
    unittest.main()
