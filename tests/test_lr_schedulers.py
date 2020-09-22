#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import parlai.nn.lr_scheduler as lr_scheduler
import parlai.utils.testing as testing_utils


class TestLRSchedulers(unittest.TestCase):
    def _run_pass(self, max_lr=1.0, warmup_updates=0, total_steps=1000, **args):
        args['warmup_updates'] = warmup_updates
        if 'max_lr_steps' not in args:
            args['max_lr_steps'] = total_steps - warmup_updates
        p = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.SGD([p], lr=max_lr)
        scheduler = lr_scheduler.ParlAILRScheduler.lr_scheduler_factory(
            args, optimizer, {}, True
        )
        output = []
        for step in range(total_steps):
            scheduler.step(step)
            output.append(scheduler.get_last_lr())
        for step, o in enumerate(output):  # noqa: B007
            assert o <= max_lr
            assert o > 0 or step == total_steps - 1
        warmup_updates = args.get('warmup_updates', 0)
        if warmup_updates > 0:
            assert output[warmup_updates - 1] == max_lr
            # no steep cliffs of > 50% of LR
            assert (output[warmup_updates] - max_lr) / max_lr < 0.5
            # LR is always linear
            for step in range(warmup_updates - 1):
                self.assertAlmostEqual(
                    output[step + 1] - output[step], max_lr / warmup_updates, places=3
                )
        return output

    def _run_resume(self, max_lr=1.0, warmup_updates=0, total_steps=200, **args):
        args['warmup_updates'] = warmup_updates
        if 'max_lr_steps' not in args:
            args['max_lr_steps'] = total_steps - warmup_updates
        p = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.SGD([p], lr=max_lr)
        scheduler = lr_scheduler.ParlAILRScheduler.lr_scheduler_factory(
            args, optimizer, {}, True
        )

        for step in range(total_steps):
            p = torch.nn.Parameter(torch.randn(4, 4))
            optimizer2 = torch.optim.SGD([p], lr=max_lr)
            sd = {
                'number_training_updates': step + 1,
                'lr_scheduler': scheduler.get_state_dict(),
                'lr_scheduler_type': args['lr_scheduler'],
                'warmup_scheduler': scheduler.get_warmup_state_dict(),
            }
            scheduler2 = lr_scheduler.ParlAILRScheduler.lr_scheduler_factory(
                args, optimizer2, sd, False
            )
            assert scheduler.get_last_lr() == scheduler2.get_last_lr(), step
            scheduler.step(step)

        sd = {
            'number_training_updates': step,
            'lr_scheduler': scheduler.get_state_dict(),
            'lr_scheduler_type': args['lr_scheduler'],
            'warmup_scheduler': scheduler.get_warmup_state_dict(),
        }
        optimizer2 = torch.optim.SGD([p], lr=max_lr)
        scheduler2 = lr_scheduler.ParlAILRScheduler.lr_scheduler_factory(
            args, optimizer2, sd, False
        )
        assert scheduler.get_last_lr() == scheduler2.get_last_lr()

    def test_cosine(self):
        self._run_pass(lr_scheduler='cosine', warmup_updates=0)
        self._run_pass(lr_scheduler='cosine', warmup_updates=50)
        with self.assertRaises(lr_scheduler.StopTrainException):
            self._run_pass(lr_scheduler='cosine', max_lr_steps=100, total_steps=1000)

    def test_linear(self):
        self._run_pass(lr_scheduler='linear', warmup_updates=0)
        self._run_pass(lr_scheduler='linear', warmup_updates=50)
        with self.assertRaises(lr_scheduler.StopTrainException):
            self._run_pass(lr_scheduler='linear', max_lr_steps=100, total_steps=1000)

    def test_invsqrt(self):
        self._run_pass(lr_scheduler='invsqrt', warmup_updates=0)
        self._run_pass(lr_scheduler='invsqrt', warmup_updates=50)

        # decay very fast
        steps = self._run_pass(
            lr_scheduler='invsqrt', warmup_updates=50, invsqrt_lr_decay_gamma=1
        )
        self.assertAlmostEquals(steps[-1], 0.0324443)

        # decay very slowly
        steps = self._run_pass(
            lr_scheduler='invsqrt', warmup_updates=50, invsqrt_lr_decay_gamma=5000
        )
        assert all(x > 0.9 for x in steps[50:])

    def test_cosine_resume(self):
        self._run_resume(lr_scheduler='cosine', warmup_updates=0)
        self._run_resume(lr_scheduler='cosine', warmup_updates=50)

    def test_linear_resume(self):
        self._run_resume(lr_scheduler='linear', warmup_updates=0)
        self._run_resume(lr_scheduler='linear', warmup_updates=50)

    def test_invsqrt_resume(self):
        self._run_resume(lr_scheduler='invsqrt', warmup_updates=0)
        self._run_resume(lr_scheduler='invsqrt', warmup_updates=50)

    def _run_end2end(
        self, lr_scheduler, max_lr=1.0, warmup_updates=0, total_steps=100, **args
    ):
        testing_utils.train_model(
            {
                'task': 'integration_tests:nocandidate',
                'model': 'test_agents/unigram',
                'skip_generation': True,
                'lr_scheduler': lr_scheduler,
                'max_lr_steps': total_steps,
                'warmup_updates': warmup_updates,
                'learningrate': max_lr,
            }
        )

    def test_end2end_cosine(self):
        self._run_end2end(lr_scheduler='cosine', warmup_updates=0)
        self._run_end2end(lr_scheduler='cosine', warmup_updates=50)

    def test_end2end_linear(self):
        self._run_end2end(lr_scheduler='linear', warmup_updates=0)
        self._run_end2end(lr_scheduler='linear', warmup_updates=50)

    def test_end2end_invsqrt(self):
        self._run_end2end(lr_scheduler='invsqrt', warmup_updates=0)
        self._run_end2end(lr_scheduler='invsqrt', warmup_updates=50)
