#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test code for anti-scaling transformer/generator models.
"""

import unittest

import numpy as np
import torch

import parlai.utils.testing as testing_utils
from parlai.core.opt import Opt
from parlai.zoo.bart.build import download as download_bart
from parlai.zoo.blender.blender_90M import download as download_blender


class TestDistillation(unittest.TestCase):
    """
    Test agents for distilling transformer/generator models.
    """

    BLENDERBOT_MODEL_FILE = 'data/models/blender/blender_90M/model'
    BART_MODEL_FILE = 'data/models/bart/bart_large/model'
    BASE_OPT = {
        'allow_missing_init_opts': True,
        'init_model': '',
        'model_file': '',
        'n_encoder_layers': 1,
        'n_decoder_layers': 1,
        'task': 'blended_skill_talk',
    }
    TRANSFORMER_OPT = {
        'init_opt': f'{BLENDERBOT_MODEL_FILE}.opt',
        'teacher_model': BLENDERBOT_MODEL_FILE,
    }
    BART_OPT = {'init_opt': f'{BART_MODEL_FILE}.opt', 'teacher_model': BART_MODEL_FILE}
    WIDE_DISTILLATION_OPT = {'copy_teacher_weights': True}
    NARROW_DISTILLATION_OPT = {'embedding_size': 64, 'ffn_size': 256}
    NARROW_DISTILLATION_DUMMY_LOSS_OPT = {
        **NARROW_DISTILLATION_OPT,
        'embedding_loss_coeff': 1,
        'self_attn_loss_coeff': 1,
        'enc_dec_attn_loss_coeff': 1,
    }
    DISTILLATION_MODEL_PREFIX = 'projects.anti_scaling.distillation'

    def setUp(self):
        """
        Download models in advance so that their opt files can be used with --init-opt
        """
        data_path = 'data'
        download_blender(data_path)
        download_bart(data_path)

    def test_distillation_losses(self):
        """
        Check the sum of distillation losses.

        Make sure that the sum of all distillation losses from one pass through the
        student and teacher models is as expected.
        """

        torch.manual_seed(0)
        np.random.seed(0)

        opts_and_desired_losses = [
            (
                self.TRANSFORMER_OPT,
                self.WIDE_DISTILLATION_OPT,
                'DistillTransformerAgent',
                {
                    'dec_hid_loss': 87.27,
                    'enc_hid_loss': 0.8726,
                    'enc_loss': 0.8726,
                    'loss': 15.85,
                    'pred_loss': 13.77,
                },
            ),
            (
                self.BART_OPT,
                self.WIDE_DISTILLATION_OPT,
                'DistillBartAgent',
                {
                    'dec_hid_loss': 2.731,
                    'enc_hid_loss': 0.0383,
                    'enc_loss': 0.0383,
                    'loss': 32.16,
                    'pred_loss': 29.35,
                },
            ),
            (
                self.TRANSFORMER_OPT,
                self.NARROW_DISTILLATION_DUMMY_LOSS_OPT,
                'DistillNarrowTransformerAgent',
                {
                    'dec_emb_loss': 1.625,
                    'dec_hid_loss': 49.08,
                    'dec_self_attn_loss': 3.609,
                    'enc_dec_attn_loss': 29.65,
                    'enc_emb_loss': 1.027,
                    'enc_hid_loss': 0.5528,
                    'enc_loss': 0.5478,
                    'enc_self_attn_loss': np.inf,
                    'loss': 11.41,
                    'pred_loss': 9.238,
                },
            ),
            (
                self.BART_OPT,
                self.NARROW_DISTILLATION_DUMMY_LOSS_OPT,
                'DistillNarrowBartAgent',
                {
                    'dec_emb_loss': 9.495,
                    'dec_hid_loss': 0.6304,
                    'dec_self_attn_loss': 513.2,
                    'enc_dec_attn_loss': 137.6,
                    'enc_emb_loss': 6.642,
                    'enc_hid_loss': 0.05015,
                    'enc_loss': 0.05089,
                    'enc_self_attn_loss': 9.318,
                    'loss': 11.39,
                    'pred_loss': 8.913,
                },
            ),
        ]
        for (
            model_opt,
            distillation_opt,
            model_name,
            desired_losses,
        ) in opts_and_desired_losses:
            opt = {
                **self.BASE_OPT,
                **model_opt,
                **distillation_opt,
                'model': f'{self.DISTILLATION_MODEL_PREFIX}:{model_name}',
                'num_examples': 1,
                'skip_generation': False,
                'hidden_loss_coeff': 1,
                'encoder_loss_coeff': 1,
                'pred_loss_coeff': 1,
                'task_loss_coeff': 1,
            }
            valid, _ = testing_utils.eval_model(Opt(opt), skip_test=True)
            for loss_name, desired_loss in desired_losses.items():
                if np.isinf(desired_loss):
                    self.assertTrue(np.isinf(valid[loss_name].value()))
                else:
                    self.assertAlmostEqual(valid[loss_name], desired_loss, delta=0.1)


if __name__ == '__main__':
    unittest.main()
