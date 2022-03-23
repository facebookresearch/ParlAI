#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
import torch
import parlai.nn.lr_scheduler as lr_scheduler
import parlai.utils.testing as testing_utils


class TestLRSchedulers(unittest.TestCase):
    def _run_pass(self, max_lr=1.0, total_steps=1000, end_zero=False, **args):
        if 'max_train_steps' not in args:
            args['max_train_steps'] = total_steps
        # for checks of correctness, hardcode warmup_rate to be 0
        args['warmup_rate'] = 0
        p = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.SGD([p], lr=max_lr)
        scheduler = lr_scheduler.ParlAILRScheduler.lr_scheduler_factory(
            args, optimizer, {}, True
        )
        output = []
        for step in range(total_steps):
            scheduler.step(step)
            output.append(scheduler.get_last_lr())
        for value in output:
            assert value >= 0
        for step, o in enumerate(output):  # noqa: B007
            assert o <= max_lr
            assert o > 0 or step == total_steps - 1
        warmup_updates = args.get('warmup_updates', 0)
        assert warmup_updates >= 0
        if warmup_updates > 0:
            assert abs(max_lr - output[warmup_updates - 1]) < 0.04
            # LR is always linear
            for step in range(warmup_updates - 2):
                self.assertAlmostEqual(
                    output[step + 1] - output[step], 1 / warmup_updates
                )
        if end_zero:
            self.assertAlmostEquals(output[-1], 0)
        else:
            self.assertNotAlmostEqual(output[-1], 0)
        return output

    def _run_resume(self, max_lr=1.0, warmup_updates=0, total_steps=200, **args):
        args['warmup_updates'] = warmup_updates
        if 'max_train_steps' not in args:
            args['max_train_steps'] = total_steps
        p = torch.nn.Parameter(torch.randn(4, 4))
        optimizer = torch.optim.SGD([p], lr=max_lr)
        scheduler = lr_scheduler.ParlAILRScheduler.lr_scheduler_factory(
            args, optimizer, {}, True
        )

        for step in range(total_steps):
            p = torch.nn.Parameter(torch.randn(4, 4))
            optimizer2 = torch.optim.SGD([p], lr=max_lr)
            sd = {
                'number_training_updates': step,
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
        self._run_pass(lr_scheduler='cosine', warmup_updates=0, end_zero=True)
        self._run_pass(lr_scheduler='cosine', warmup_updates=50, end_zero=True)
        with self.assertRaises(lr_scheduler.StopTrainException):
            self._run_pass(lr_scheduler='cosine', max_train_steps=100, total_steps=1000)

    def test_linear(self):
        self._run_pass(lr_scheduler='linear', warmup_updates=0, end_zero=True)
        self._run_pass(lr_scheduler='linear', warmup_updates=50, end_zero=True)
        with self.assertRaises(lr_scheduler.StopTrainException):
            self._run_pass(lr_scheduler='linear', max_train_steps=100, total_steps=1000)

    def test_invsqrt(self):
        self._run_pass(lr_scheduler='invsqrt', warmup_updates=0, end_zero=False)
        self._run_pass(lr_scheduler='invsqrt', warmup_updates=50, end_zero=False)

        # decay very fast
        steps = self._run_pass(
            lr_scheduler='invsqrt',
            warmup_updates=50,
            invsqrt_lr_decay_gamma=1,
            end_zero=False,
        )
        self.assertAlmostEquals(steps[-1], 0.03242722)

        # decay very slowly
        steps = self._run_pass(
            lr_scheduler='invsqrt',
            warmup_updates=50,
            invsqrt_lr_decay_gamma=5000,
            end_zero=False,
        )
        assert all(x > 0.9 for x in steps[50:])

    def test_cosine_resume(self):
        self._run_resume(lr_scheduler='cosine', warmup_updates=0, end_zero=True)
        self._run_resume(lr_scheduler='cosine', warmup_updates=50, end_zero=True)

    def test_linear_resume(self):
        self._run_resume(lr_scheduler='linear', warmup_updates=0, end_zero=True)
        self._run_resume(lr_scheduler='linear', warmup_updates=50, end_zero=True)

    def test_invsqrt_resume(self):
        self._run_resume(lr_scheduler='invsqrt', warmup_updates=0, end_zero=True)
        self._run_resume(lr_scheduler='invsqrt', warmup_updates=50, end_zero=True)

    def _run_end2end(
        self, lr_scheduler, max_lr=1.0, warmup_updates=0, total_steps=100, **args
    ):
        testing_utils.train_model(
            {
                'task': 'integration_tests:nocandidate',
                'model': 'test_agents/unigram',
                'skip_generation': True,
                'lr_scheduler': lr_scheduler,
                'max_train_steps': total_steps,
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


class TestLRIntegration(unittest.TestCase):
    """
    Deep LR scheduler tests to check how we handle preemption.
    """

    PREEMPT = 30

    def _test_scheduler(self, **kwargs):
        from parlai.scripts.train_model import TrainModel, TrainLoop

        # shallow copy to prevent overwrites
        kwargs = kwargs.copy()
        with testing_utils.tempdir() as tmpdir:
            kwargs['model'] = 'test_agents/unigram'
            kwargs['task'] = 'integration_tests'
            kwargs['skip_generation'] = True
            kwargs['validation_metric'] = 'loss'
            kwargs['model_file'] = os.path.join(tmpdir, 'model')
            kwargs['dict_file'] = 'zoo:unittest/transformer_generator2/model.dict'
            kwargs['log_every_n_steps'] = 1
            kwargs['validation_every_n_steps'] = 10
            kwargs['max_train_steps'] = 100
            kwargs['save_after_valid'] = True
            kwargs['learningrate'] = 1
            opt = TrainModel.setup_args().parse_kwargs(**kwargs)

            logs_first = []
            for i, train_step_log in enumerate(TrainLoop(opt).train_steps(), 1):
                logs_first.append(train_step_log)
                if i >= self.PREEMPT - 2:
                    # simulate preemption
                    break

            # resume training
            logs_second = []
            for train_step_log in TrainLoop(opt).train_steps():
                logs_second.append(train_step_log)

            # check correctness
            assert (
                logs_first[20]['total_train_updates']
                == logs_second[0]['total_train_updates']
            )
            assert logs_first[20]['lr'] == logs_second[0]['lr']

            if 'warmup_updates' in kwargs:
                full_logs = logs_first[:20] + logs_second
                assert abs(1.0 - full_logs[kwargs['warmup_updates'] - 1]['lr']) < 0.04

            return logs_first, logs_second

    def test_invsqrt(self):
        self._test_scheduler(lr_scheduler='invsqrt')

    def test_invsqrt_warmup(self):
        self._test_scheduler(lr_scheduler='invsqrt', warmup_updates=25)

    def test_invsqrt_long_warmup(self):
        self._test_scheduler(lr_scheduler='invsqrt', warmup_updates=self.PREEMPT + 30)

    def test_reduceonplateau(self):
        self._test_scheduler(lr_scheduler='reduceonplateau')

    def test_reduceonplateau_warmup(self):
        self._test_scheduler(lr_scheduler='reduceonplateau', warmup_updates=25)

    def test_reduceonplateau_long_warmup(self):
        self._test_scheduler(
            lr_scheduler='reduceonplateau', warmup_updates=self.PREEMPT + 30
        )

    def test_linear(self):
        self._test_scheduler(lr_scheduler='linear')

    def test_linear_warmup(self):
        self._test_scheduler(lr_scheduler='linear', warmup_updates=25)

    def test_linear_long_warmup(self):
        self._test_scheduler(lr_scheduler='linear', warmup_updates=self.PREEMPT + 30)

    def test_cosine(self):
        self._test_scheduler(lr_scheduler='cosine')

    def test_cosine_warmup(self):
        self._test_scheduler(lr_scheduler='cosine', warmup_updates=25)

    def test_cosine_long_warmup(self):
        self._test_scheduler(lr_scheduler='cosine', warmup_updates=self.PREEMPT + 30)
