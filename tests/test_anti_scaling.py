#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Test code for anti-scaling transformer/generator models.
"""

import unittest

import parlai.utils.testing as testing_utils


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
        blenderbot_model_file = 'zoo:blender/blender_90M/model'
        bart_model_file = 'zoo:bart/bart_large/model'
        base_opt = {
            'init_model': None,
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
            (transformer_opt, wide_distillation_opt, 'DistillTransformerAgent', 0),
            (bart_opt, wide_distillation_opt, 'DistillBartAgent', 0),
            (
                transformer_opt,
                narrow_distillation_opt,
                'DistillNarrowTransformerAgent',
                0,
            ),
            (bart_opt, narrow_distillation_opt, 'DistillNarrowBartAgent', 0),
        ]  # TODO: change losses
        for (
            model_opt,
            distillation_opt,
            model_name,
            desired_loss,
        ) in opts_and_desired_losses:
            opt = {
                **base_opt,
                **model_opt,
                **distillation_opt,
                'model': f'projects.anti_scaling.distillation:{model_name}',
            }
            valid, _ = testing_utils.eval_model(opt, skip_test=True)
            self.assertAlmostEqual(
                valid['loss'], desired_loss, delta=0
            )  # TODO: change delta
