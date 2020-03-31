#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest
import torch
import parlai.nn.lr_scheduler as lr_scheduler


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
            output.append(optimizer.param_groups[0]['lr'])
        for step, o in enumerate(output):  # noqa: B007
            assert o <= max_lr
            assert o > 0
        warmup_updates = args.get('warmup_updates', 0)
        if warmup_updates > 0:
            assert output[warmup_updates] == max_lr
            # no steep cliffs of > 50% of LR
            assert (output[warmup_updates + 1] - max_lr) / max_lr < 0.5
            # LR is always linear
            for step in range(warmup_updates - 1):
                self.assertAlmostEqual(
                    output[step + 1] - output[step], max_lr / warmup_updates, places=3,
                )
        return output

    def test_cosine(self):
        self._run_pass(lr_scheduler='cosine', warmup_updates=0)
        self._run_pass(lr_scheduler='cosine', warmup_updates=50)
        with self.assertRaises(lr_scheduler.StopTrainException):
            self._run_pass(lr_scheduler='cosine', max_lr_steps=100, total_steps=1000)

    def test_linear_warmup(self):
        self._run_pass(lr_scheduler='linear', warmup_updates=0)
        self._run_pass(lr_scheduler='linear', warmup_updates=50)
        with self.assertRaises(lr_scheduler.StopTrainException):
            self._run_pass(lr_scheduler='cosine', max_lr_steps=100, total_steps=1000)

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
