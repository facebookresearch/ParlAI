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
from parlai.zoo.bart.build import download as download_bart
from parlai.zoo.blender.blender_90M import download as download_blender


class TestDistillation(unittest.TestCase):
    """
    Test agents for distilling transformer/generator models.
    """

    def test_distillation_losses(self):
        """
        Check the sum of distillation losses.

        Make sure that the sum of all distillation losses from one pass through the
        student and teacher models is as expected.
        """

        torch.manual_seed(0)
        np.random.seed(0)

        # Download models in advance so that their opt files can be used with --init-opt
        data_path = 'data'
        download_blender(data_path)
        download_bart(data_path)
        blenderbot_model_file = 'data/models/blender/blender_90M/model'
        bart_model_file = 'data/models/bart/bart_large/model'

        base_opt = {
            'allow_missing_init_opts': True,
            'init_model': '',
            'model_file': '',
            'n_encoder_layers': 1,
            'n_decoder_layers': 1,
            'num_examples': 1,
            'skip_generation': False,
            'task': 'blended_skill_talk',
            'hidden_loss_coeff': 1,
            'encoder_loss_coeff': 1,
            'pred_loss_coeff': 1,
            'task_loss_coeff': 1,
        }
        transformer_opt = {
            'init_opt': f'{blenderbot_model_file}.opt',
            'teacher_model': blenderbot_model_file,
        }
        bart_opt = {
            'init_opt': f'{bart_model_file}.opt',
            'teacher_model': bart_model_file,
        }
        wide_distillation_opt = {'copy_teacher_weights': True}
        narrow_distillation_opt = {
            'embedding_size': 64,
            'ffn_size': 256,
            'embedding_loss_coeff': 1,
            'self_attn_loss_coeff': 1,
            'enc_dec_attn_loss_coeff': 1,
        }
        opts_and_desired_losses = [
            (
                transformer_opt,
                wide_distillation_opt,
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
                bart_opt,
                wide_distillation_opt,
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
                transformer_opt,
                narrow_distillation_opt,
                'DistillNarrowTransformerAgent',
                {
                    'dec_emb_loss': 0.1245,
                    'dec_hid_loss': 49.08,
                    'dec_self_attn_loss': 3.609,
                    'enc_dec_attn_loss': 29.65,
                    'enc_emb_loss': 0.07892,
                    'enc_hid_loss': 0.5528,
                    'enc_loss': 0.5478,
                    'enc_self_attn_loss': np.inf,
                    'loss': 11.41,
                    'pred_loss': 9.238,
                },
            ),
            (
                bart_opt,
                narrow_distillation_opt,
                'DistillNarrowBartAgent',
                {
                    'dec_emb_loss': 0.1323,
                    'dec_hid_loss': 0.6304,
                    'dec_self_attn_loss': 513.2,
                    'enc_dec_attn_loss': 137.6,
                    'enc_emb_loss': 0.09919,
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
                **base_opt,
                **model_opt,
                **distillation_opt,
                'model': f'projects.anti_scaling.distillation:{model_name}',
            }
            valid, _ = testing_utils.eval_model(opt, skip_test=True)
            for loss_name, desired_loss in desired_losses.items():
                if np.isinf(desired_loss):
                    self.assertTrue(np.isinf(valid[loss_name].value()))
                else:
                    self.assertAlmostEqual(valid[loss_name], desired_loss, delta=0.1)


if __name__ == '__main__':
    unittest.main()
